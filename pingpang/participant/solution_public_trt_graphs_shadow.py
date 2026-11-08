"""Diagnostic only: compare both event backends on every actual window."""
import torch
from participant.solution_public_trt_graphs import Solution as EventSolution

class Solution(EventSolution):
    def prepare(self,ctx):
        super().prepare(ctx);self.shadow_rows=[];ctx.info['event_shadow']=self.shadow_rows
        deployed=self.system.head.forward;reference=self.original_event_forward
        def forward(*args):
            got=deployed(*args);ref=reference(*args)
            a=ref.float().sigmoid();b=got.float().sigmoid()
            self.shadow_rows.append(dict(video_id=self.vid,frame_id=self.next_center,
                logits_max_abs=float((got.float()-ref.float()).abs().max()),probability_max_abs=float((a-b).abs().max()),
                reference=a[0].tolist(),candidate=b[0].tolist(),threshold_flip=bool((a[0,0]>=self.threshold)!=(b[0,0]>=self.threshold))))
            return got
        self.system.head.forward=forward

