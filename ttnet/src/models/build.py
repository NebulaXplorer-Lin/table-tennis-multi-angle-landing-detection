"""Component constructors keep checkpoint tensor names independent of presets."""
from ttnet.temporal import temporal_mode
from .localizer import BallLocalizer
from .event import EventHead, TransformerEventHead


def build_localizer(config=None):
    config = config or {}
    if config.get('localizer_type','residual_fpn') != 'residual_fpn':
        raise ValueError('unsupported localizer_type')
    return BallLocalizer(temporal_mode(config))


def build_event_head(config=None):
    config = config or {}
    kind = config.get('event_head_type','tcn')
    if kind == 'tcn':return EventHead()
    if kind == 'transformer':return TransformerEventHead(**config.get('transformer',{}))
    raise ValueError('unsupported event_head_type')
