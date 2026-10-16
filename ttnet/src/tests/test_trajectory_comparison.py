import unittest

from ttnet.evaluation.trajectory_comparison import measure


class TrajectoryComparisonTests(unittest.TestCase):
    def test_error_runs_jumps_and_roi_use_observations_only(self):
        rows=[]; records=[]
        for fid, x in enumerate([0, 10, 20, 30, 40, 50]):
            rows.append(dict(video_id='v', frame_id=fid, timestamp_sec=fid*.04, fps=25,
                label_status='positive', ball_centers=[[x,20]], resolution=dict(width=1000,height=100)))
            pred=None if fid in (2,3) else dict(x=x if fid!=4 else 500, y=20, confidence=.9, valid=True)
            records.append(dict(video_id='v', frame_id=fid, prediction=pred,
                crop_transform=None, roi_transforms=[dict(left=0,top=0,side=100)],
                search_prediction=dict(x=x,y=20,valid=False,predicted=True)))
        result, jumps=measure(rows, records, {'v':[3]})
        self.assertEqual(result['localization']['50']['tp'],3)
        self.assertEqual(result['consecutive_missed']['50']['count'],1)
        self.assertEqual(result['consecutive_missed']['50']['max'],3)
        self.assertEqual(result['consecutive_no_observation']['max'],2)
        self.assertEqual(result['roi_any_coverage'],1.)
        self.assertEqual(result['roi_selected_coverage'],0.)
        self.assertEqual(result['bounce_neighborhood']['frames'],3)
        self.assertEqual(len(jumps),1)
        self.assertEqual(jumps[0]['frame_id'],5)

    def test_missing_unknown_and_video_boundary_break_runs_and_jumps(self):
        rows=[]; records=[]
        for vid, fid, status in [('v',0,'positive'),('v',1,'unknown'),('v',3,'positive'),('w',0,'positive')]:
            rows.append(dict(video_id=vid,frame_id=fid,timestamp_sec=fid*.04,fps=25,
                label_status=status,ball_centers=[[10,10]] if status=='positive' else [],resolution=dict(width=100,height=100)))
            records.append(dict(video_id=vid,frame_id=fid,prediction=None,roi_transforms=[]))
        result,jumps=measure(rows,records,{})
        self.assertEqual(result['consecutive_no_observation']['count'],3)
        self.assertEqual(result['consecutive_no_observation']['max'],1)
        self.assertEqual(result['erroneous_jumps']['accepted_adjacent_pairs'],0)
        self.assertEqual(jumps,[])


if __name__ == '__main__':unittest.main()
