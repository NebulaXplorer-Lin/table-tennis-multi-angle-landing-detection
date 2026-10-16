"""Regression checks for coarse-event import without changing the ball split."""
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np

from ttnet.common import identity, jsonl, save, save_jsonl, sha
from ttnet.data_tools.frame_store import FrameStore, pack_video
from ttnet.data_tools.merge_pp_events import (frame_annotation, label_tag, merge, parse_xml, reconcile,
                                             verify, video_id_for)
from ttnet.training.dataset import BallDataset


def xml(name, tags=(), count=4, image_entries=True):
    images=''.join(f'<image id="{i}" name="frame_{i:06d}" width="64" height="32">'+
        ''.join(f'<tag label="{label}" source="manual"/>' for frame,label in tags if frame==i)+'</image>' for i in range(count)) if image_entries else ''
    return (f'<annotations><version>1.1</version><meta><task><id>7</id><name>{name}</name><source>{name[-3:]}.mp4</source>'
        f'<size>{count}</size><start_frame>0</start_frame><stop_frame>{count-1}</stop_frame><frame_filter>step=1</frame_filter>'
        '<original_size><width>64</width><height>32</height></original_size></task></meta>'+images+'</annotations>').encode()


class MergeEventsTests(unittest.TestCase):
    def test_aliases_and_mask_policy(self):
        for name in ('table_contact','table_bounce','touch_table','table'):
            event=frame_annotation([label_tag(0,name)])
            self.assertEqual(event['target'],[1.,None]);self.assertEqual(event['supervised'],[True,False])
        for name in ('net_contact','net hit','touch_net','net'):
            event=frame_annotation([label_tag(0,name)])
            self.assertEqual(event['status'],['unknown','positive']);self.assertFalse(any(event['supervised']))
        for name in ('racket_hit','floor_bounce'):
            event=frame_annotation([label_tag(0,name)])
            self.assertEqual(event['target'],[0.,None]);self.assertEqual(event['supervised'],[True,False])
        unknown=frame_annotation([])
        self.assertEqual(unknown['target'],[None,None]);self.assertFalse(any(unknown['supervised']))
        ignored=frame_annotation([label_tag(0,'ignore'),label_tag(0,'racket_hit')])
        self.assertEqual(ignored['status'],['ignore','ignore']);self.assertFalse(any(ignored['supervised']))
        conflict=frame_annotation([label_tag(0,'table'),label_tag(0,'racket_hit')])
        self.assertEqual(conflict['status'][0],'conflict');self.assertFalse(any(conflict['supervised']))
        with self.assertRaises(ValueError):label_tag(0,'unknown_new_label')
        with self.assertRaises(ValueError):label_tag(-1,'table')

    def test_task_ids_never_guess_video_names(self):
        self.assertEqual(video_id_for(Path('/source/01/012673452.zip'),'01001','001.mp4'),'01_001')
        self.assertEqual(video_id_for(Path('/source/01/dataset_task_1.zip'),'01007.mp4','007.mp4'),'01_007')
        self.assertEqual(video_id_for(Path('/source/02/02_007_events/annotations.xml')),'02_007')
        with self.assertRaises(ValueError):video_id_for(Path('/source/task42.zip'))
        with self.assertRaises(ValueError):video_id_for(Path('/source/02/02_007_events.zip'),'03_007')

    def test_empty_exports_and_conflicting_duplicates(self):
        parsed=parse_xml(xml('02004',image_entries=False),Path('/source/02/02_004_events.zip'))
        parsed['provenance']={'path':'a'}
        meta=dict(actual_frames=4,resolution=dict(width=64,height=32),video_sha256='hash')
        tags,audit=reconcile('02_004',[parsed],meta)
        self.assertEqual(tags,[]);self.assertIn('empty_event_export_all_frames_unknown',audit['warnings'])
        other=dict(parsed,tags=[label_tag(1,'table')])
        with self.assertRaises(ValueError):reconcile('02_004',[parsed,other],meta)
        with self.assertRaises(ValueError):reconcile('02_004',[dict(parsed,frame_count=5)],meta)
        with self.assertRaises(ValueError):reconcile('02_004',[dict(parsed,source_sha256='wrong')],meta)

    def fixture(self,root):
        base=root/'ball';source=root/'annotations';output=root/'merged';store=root/'store'
        entries=[];videos={};split_hashes={};tag_sets={}
        for part,vid in [('train','00_001'),('val','01_001')]:
            original=root/f'{vid}.mp4';original.write_bytes(vid.encode())
            entry=dict(video_id=vid,group=vid[:2],split=part,video=original.name,video_sha256=sha(original))
            save_jsonl(base/f'splits/{part}.jsonl',[entry]);split_hashes[part]=sha(base/f'splits/{part}.jsonl')
            rows=[dict(video_id=vid,split=part,frame_id=i,timestamp_sec=i/25,frame_path=f'frames/{vid}/{i:06d}.png',
                resolution=dict(width=64,height=32),label_status='positive',ball_centers=[[20.,10.]],ball_boxes=[[18.,8.,22.,12.]]) for i in range(4)]
            save_jsonl(base/f'labels/{vid}.jsonl',rows);save_jsonl(base/f'labels/{part}.jsonl',rows)
            save_jsonl(base/f'windows/{vid}.jsonl',[dict(frame_id=i,unchanged=True) for i in range(4)])
            save(base/f'metadata/videos/{vid}.json',dict(actual_frames=4,resolution=dict(width=64,height=32),video_sha256=sha(original),
                labels_sha256=sha(base/f'labels/{vid}.jsonl'),windows_sha256=sha(base/f'windows/{vid}.jsonl')))
            manifest=pack_video(store,vid,rows,lambda r:np.full((32,64,3),r['frame_id'],np.uint8))
            videos[vid]=manifest;entries.append(entry)
            tags=[(1,'table_contact'),(1,'table_contact'),(2,'ignore'),(2,'racket_hit'),(3,'floor_bounce')] if part=='train' else [(1,'net')]
            tag_sets[vid]=tags
            dest=source/vid[:2]/f'{vid}_events.zip';dest.parent.mkdir(parents=True)
            with zipfile.ZipFile(dest,'w') as archive:archive.writestr('annotations.xml',xml(vid,tags))
            if part=='train':
                duplicate=dest.with_suffix('')/'annotations.xml';duplicate.parent.mkdir();duplicate.write_bytes(xml(vid,tags))
        save(store/'manifest.json',dict(status='completed',storage_id='test',videos=videos))
        save(base/'frame_store.json',dict(storage_id='test',path='../store'))
        save(base/'metadata/dataset.json',dict(status='completed',dataset_id='base',split_hashes=split_hashes,
            frames=8,source_videos=entries))
        return base,source,output

    def test_complete_merge_reuses_arrays_and_keeps_positions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base,source,output=self.fixture(root)
            hashes={str(p):sha(p) for p in base.rglob('*') if p.is_file()}
            report=merge(source,base,output,project=root)
            check=verify(output,base)
            self.assertEqual(check['frames'],8);self.assertTrue(check['frame_store_reused'])
            self.assertEqual(report['duplicate_representations'],1)
            self.assertEqual(report['per_split']['train']['event_counts']['bounce_positive'],1)
            self.assertEqual(report['per_split']['train']['event_counts']['bounce_negative'],1)
            self.assertEqual(report['per_split']['train']['event_counts']['bounce_ignore'],1)
            self.assertEqual(report['per_split']['val']['event_counts']['net_supervised'],0)
            self.assertEqual(hashes,{str(p):sha(p) for p in base.rglob('*') if p.is_file()})
            np.testing.assert_array_equal(FrameStore(output).get_rgb('00_001',1),FrameStore(base).get_rgb('00_001',1))
            a=BallDataset(output,'train',temporal_mode='fixed3',input_size=[64,32])[1]
            b=BallDataset(base,'train',temporal_mode='fixed3',input_size=[64,32])[1]
            for key in a[0]:np.testing.assert_array_equal(a[0][key].numpy(),b[0][key].numpy())
            np.testing.assert_array_equal(a[1].numpy(),b[1].numpy())
            with self.assertRaises(FileExistsError):merge(source,base,output,project=root)
            for event in jsonl(output/'events/train.jsonl'):self.assertIsNone(event['contact_xy'])

    def test_changed_source_and_invalid_frame_abort_before_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base,source,output=self.fixture(root)
            (root/'00_001.mp4').write_bytes(b'changed')
            with self.assertRaises(ValueError):merge(source,base,output,project=root)
            self.assertFalse(output.exists())

    def test_json_matches_both_events_and_cvat_tags(self):
        from ttnet.data_tools.merge_pp_events import parse_json
        doc=dict(video_id='13_001',task_name='TableTennis_13_001',frame_index_base=0,frame_count=4,
            source_sha256='hash',labels=[dict(id=1,name='table_bounce')],events=[dict(frame_id=1,type='table_bounce')],
            data_meta=dict(size=4,start_frame=0,stop_frame=3,frame_filter='step=1',deleted_frames=[],frames=[dict(width=64,height=32)]),
            cvat_annotations=dict(tags=[dict(frame=1,label_id=1)],shapes=[],tracks=[],intervals=[]))
        parsed=parse_json(doc,Path('/source/13/001.json'))
        self.assertEqual(parsed['tags'][0]['kind'],'bounce')
        doc['events'][0]['frame_id']=2
        with self.assertRaises(ValueError):parse_json(doc,Path('/source/13/001.json'))


if __name__=='__main__':unittest.main()
