"""Component constructors keep checkpoint tensor names independent of presets."""
from ttnet.temporal import temporal_mode
from .localizer import BallLocalizer
from .event import EventHead, TransformerEventHead, ContextEventHead


def build_localizer(config=None, stage=None):
    config = config or {}
    stage = stage or config.get('stage')
    kind = config.get(f'{stage}_localizer_type', config.get('localizer_type', 'residual_fpn'))
    if kind == 'high_resolution_se':
        from .high_resolution import HighResolutionSELocalizer
        return HighResolutionSELocalizer(temporal_mode(config))
    if kind != 'residual_fpn':
        raise ValueError(f'unsupported localizer_type: {kind}')
    return BallLocalizer(temporal_mode(config))


def build_event_head(config=None):
    config = config or {}
    kind = config.get('event_head_type','tcn')
    if config.get('table_fusion'):
        from .event import TableFusionEventHead
        spec = config['table_fusion']
        return TableFusionEventHead(spec['channels'],spec['mode'],config.get('head_dropout',.1),config.get('time_encoding',True))
    if config.get('context_head'):
        return ContextEventHead(kind,config.get('scene_mode','none'),config.get('head_dropout',.1),config.get('time_encoding',False))
    if kind == 'tcn':return EventHead()
    if kind == 'transformer':return TransformerEventHead(**config.get('transformer',{}))
    raise ValueError('unsupported event_head_type')
