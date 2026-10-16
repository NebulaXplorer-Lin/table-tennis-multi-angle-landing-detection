"""Component constructors keep checkpoint tensor names independent of presets."""
from ttnet.temporal import temporal_mode
from .localizer import BallLocalizer
from .event import EventHead, TransformerEventHead, ContextEventHead


def build_localizer(config=None):
    config = config or {}
    if config.get('localizer_type','residual_fpn') != 'residual_fpn':
        raise ValueError('unsupported localizer_type')
    return BallLocalizer(temporal_mode(config))


def build_event_head(config=None):
    config = config or {}
    kind = config.get('event_head_type','tcn')
    if config.get('context_head'):
        return ContextEventHead(kind,config.get('scene_mode','none'),config.get('head_dropout',.1),config.get('time_encoding',False))
    if kind == 'tcn':return EventHead()
    if kind == 'transformer':return TransformerEventHead(**config.get('transformer',{}))
    raise ValueError('unsupported event_head_type')
