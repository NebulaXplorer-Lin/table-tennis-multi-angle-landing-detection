"""Revision safety: preserve other tasks, unknown masks and frozen video splits."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.build_table_reannotations import apply_revisions
from ttnet.scene import table


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TableRevisionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base, self.revised = self.root/'base', self.root/'revised'
        for root in (self.base,self.revised):
            (root/'splits').mkdir(parents=True)
            (root/'labels').mkdir()
        (self.revised/'metadata').mkdir()
        self.docs = {}
        for part in ('train','val'):
            vid = part+'_video'
            split = json.dumps(dict(video_id=vid,annotation=part+'.json'))+'\n'
            (self.base/f'splits/{part}.jsonl').write_text(split)
            (self.revised/f'splits/{part}.jsonl').write_text(split)
            doc = dict(video_id=vid,frame_width=64,frame_height=32,frames=[
                dict(frame_id=0,annotations=[dict(label='ball',bbox_xyxy=[[9,9,12,12]]),dict(label='table',bbox_xyxy=[[1,2,20,10]])]),
                dict(frame_id=1,annotations=[])])
            self.docs[part] = doc
            (self.root/f'{part}.json').write_text(json.dumps(doc))
            records = [dict(video_id=vid,frame_id=i,split=part,resolution=dict(width=64,height=32),
                            table_status='positive' if i==0 else 'unknown',
                            table_bbox_xyxy=[2,3,22,12] if i==0 else None) for i in (0,1)]
            (self.revised/f'labels/{part}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        self.refresh_meta()
        self.addCleanup(patch.stopall)
        patch.object(table,'DATA',self.base).start()
        patch.object(table,'PROJECT',self.root).start()

    def refresh_meta(self):
        self.meta = dict(schema='pp_table_dataset_v1',status='completed',dataset_id='revision-1',
                         split_hashes={p:digest(self.revised/f'splits/{p}.jsonl') for p in ('train','val')},
                         label_hashes={p:digest(self.revised/f'labels/{p}.jsonl') for p in ('train','val')})
        (self.revised/'metadata/dataset.json').write_text(json.dumps(self.meta))

    def test_explicit_labels_preserve_unknown_and_legacy_default(self):
        legacy,a=table.annotations('val')
        revised,b=table.annotations('val',dataset=self.revised)
        self.assertEqual(legacy[('val_video',0)],[1,2,20,10])
        self.assertEqual(revised[('val_video',0)],[2,3,22,12])
        self.assertNotIn(('val_video',1),revised)
        self.assertEqual(a,b)
        self.assertEqual(b['missing'],1)

    def test_relabeling_preserves_ball_and_original_document(self):
        original=deepcopy(self.docs['val'])
        revision=dict(frame_id=0,old_bbox_xyxy=[1,2,20,10],new_bbox_xyxy=[2,3,22,12])
        changed=apply_revisions(original,[revision])
        self.assertEqual(original,self.docs['val'])
        self.assertEqual(changed['frames'][0]['annotations'][0],original['frames'][0]['annotations'][0])
        self.assertEqual(changed['frames'][1],original['frames'][1])
        self.assertEqual(changed['frames'][0]['annotations'][1]['bbox_xyxy'],[[2,3,22,12]])
        with self.assertRaises(ValueError):
            apply_revisions(original,[dict(revision,old_bbox_xyxy=[0,0,1,1])])
        with self.assertRaises(ValueError):
            apply_revisions(original,[dict(revision,new_bbox_xyxy=[-1,3,22,12])])

    def test_changed_label_file_is_rejected_before_training(self):
        path=self.revised/'labels/val.jsonl'
        path.write_text(path.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'label hash mismatch'):
            table.annotations('val',dataset=self.revised)

    def test_split_change_cannot_move_validation_video_into_training(self):
        (self.revised/'splits/train.jsonl').write_text((self.revised/'splits/val.jsonl').read_text())
        self.refresh_meta()
        with self.assertRaisesRegex(ValueError,'differs from frozen split'):
            table.annotation_identity(self.revised)

    def test_unknown_bbox_and_duplicate_frame_are_rejected(self):
        path=self.revised/'labels/val.jsonl'
        rows=[json.loads(s) for s in path.read_text().splitlines()]
        rows[1]['table_bbox_xyxy']=[1,2,20,10]
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        self.refresh_meta()
        with self.assertRaisesRegex(ValueError,'invalid revised'):
            table.annotations('val',dataset=self.revised)
        rows[1]=rows[0]
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        self.refresh_meta()
        with self.assertRaisesRegex(ValueError,'duplicate'):
            table.annotations('val',dataset=self.revised)


if __name__=='__main__':
    unittest.main()
