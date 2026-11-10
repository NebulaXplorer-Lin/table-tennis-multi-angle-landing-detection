"""Global/table/event graph replay; Local keeps independent per-crop outputs."""
import json,torch
from pathlib import Path
from participant.solution_public_trt_event import Solution as EventSolution
from participant.public_global_graph import GlobalGraph

class Solution(EventSolution):
    def prepare(self,ctx):
        super().prepare(ctx)
        manifest=json.loads((Path(ctx.info['trt_directory'])/'manifest.json').read_text())
        self.graph_backends={}
        for stage,model in [('global',self.system.global_model),('table',self.system.table_encoder)]:
            graph=GlobalGraph(self.trt_engines[stage]);self.graph_backends[stage]=graph
            dtypes=[getattr(torch,t) for t in manifest[stage]['output_dtypes']]
            def forward(*args,_graph=graph,_dtypes=dtypes):return tuple(x.to(dtype) for x,dtype in zip(_graph(*args),_dtypes))
            model.forward=forward
        event=json.loads((Path(ctx.info['event_trt_directory'])/'manifest.json').read_text())['event']
        graph=GlobalGraph(self.trt_engines['event']);self.graph_backends['event']=graph
        dtype=getattr(torch,event['output_dtype'])
        def forward(features,motion,frame_valid,table):
            return graph(features,motion,frame_valid,*(table[k] for k in ('roi','context','roi_valid','context_valid')))[0].to(dtype)
        self.system.head.forward=forward
        ctx.info['cuda_graph_stages']=['global','table','event']

        # Candidate 1: apply after engine setup, before any video/tracker reset.
        self.config.update(global_threshold=0.7, local_threshold=0.7)
        self.config['association'] = {
            **self.config.get('association', {}),
            'motion_weight': 0.8, 'motion_scale_px': 100.0, 'missing_cost': 3.0,
        }
        self.system.config = self.config
        self.threshold = 0.3
        self.nms_sec = 0.06
        ctx.info.update(parameter_profile='candidate1', actual_config=self.config,
                        threshold=self.threshold, nms_sec=self.nms_sec)
