"""Explicit JSON experiment composition; legacy flat JSON configs stay readable."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from ttnet.common import PROJECT, identity

COMPONENT_KEYS = {'localizer','tracker','event_head','training'}
EXPERIMENT_KEYS = {'schema_version','name','task','components','dataset','parameters','description'}
EXTRA_PARAMETERS = {'context_head','scene_mode','head_dropout','time_encoding','event_window_mode','table_checkpoint','table_source','ar_checkpoint','ar_beam_size','ar_use_table','mode','localizer_type','tracker_type','event_head_type','local_crop_side_1080',
                    'feature_batch','validation_batch','association','transformer','pp_annotations','sampling_protocol','local_crop_cache','global_train_scales','local_train_scales','data','dataset','global_checkpoint','local_checkpoint'}
EXTRA_PARAMETERS |= {'global_localizer_type', 'local_localizer_type', 'source_sampling'}
EXTRA_PARAMETERS |= {'global_threshold','local_threshold','table_frontend','table_fusion',
                     'table_geometry_version','motion_max_step_fraction','table_pool'}
EXTRA_PARAMETERS |= {'public_validation_root'}
PATH_KEYS = {'global_checkpoint','local_checkpoint','data','dataset','pp_annotations'}
PATH_KEYS |= {'public_validation_root'}
ALIASES = {'localizer.type':'localizer_type','localizer.temporal_mode':'temporal_mode',
           'tracker.type':'tracker_type','event_head.type':'event_head_type'}


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT/path


def merge(base, override, *, strict=False):
    result = copy.deepcopy(base)
    for key,value in override.items():
        if strict and key not in result:
            raise ValueError(f'unknown configuration key: {key}')
        if isinstance(value,dict) and isinstance(result.get(key),dict):
            result[key] = merge(result[key],value,strict=strict)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _known_parameters():
    result = {}
    for name in ('default','event'):
        result.update(json.loads((PROJECT/f'ttnet/src/configs/{name}.json').read_text()))
    return set(result)|EXTRA_PARAMETERS


def _validate_parameters(parameters):
    unknown = set(parameters)-_known_parameters()
    if unknown:
        raise ValueError(f'unknown configuration keys: {sorted(unknown)}')
    known={}
    for name in ('default','event'):
        known.update(json.loads((PROJECT/f'ttnet/src/configs/{name}.json').read_text()))
    for key,value in parameters.items():
        if isinstance(value,dict) and isinstance(known.get(key),dict):
            merge(known[key],value,strict=True)


def load_config(path, overrides=()):
    path = project_path(path)
    document = json.loads(path.read_text())
    if 'schema_version' not in document:
        # Old checkpoint/config identities must not acquire injected defaults.
        if overrides:
            raise ValueError('use a composed experiment config for --set overrides')
        return document,dict(legacy=True,sources=[str(path)],config_hash=identity(document))
    unknown = set(document)-EXPERIMENT_KEYS
    if unknown or document['schema_version'] != 1:
        raise ValueError(f'unsupported experiment schema or keys: {sorted(unknown)}')
    if document['task'] not in ('ball','event'):
        raise ValueError('task must be ball or event')
    components = document['components']
    if set(components)-COMPONENT_KEYS:
        raise ValueError('unknown component group')
    config = {}
    sources = [str(path)]
    for kind,relative in components.items():
        source = (path.parent/relative).resolve()
        values = json.loads(source.read_text())
        _validate_parameters(values)
        config = merge(config,values)
        sources.append(str(source))
    dataset_path = (path.parent/document['dataset']).resolve()
    dataset = json.loads(dataset_path.read_text())
    if set(dataset) != {'name','kind','path','split_id'}:
        raise ValueError('dataset needs name, kind, path and split_id')
    if dataset['kind'] != document['task']:
        raise ValueError('dataset kind differs from task')
    sources.append(str(dataset_path))
    parameters = document.get('parameters',{})
    _validate_parameters(parameters)
    # Nested parameter objects must have known fields too, e.g. early stopping.
    for key,value in parameters.items():
        if key not in config:config[key]=copy.deepcopy(value)
        else:config=merge(config,{key:value},strict=True)
    for override in overrides:
        if '=' not in override:
            raise ValueError('--set requires key=value')
        key,raw = override.split('=',1)
        key = ALIASES.get(key,key)
        if key.startswith('training.'):
            key = key[len('training.'):]
        parts = key.split('.')
        target = config
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part],dict):
                raise ValueError(f'unknown configuration key: {key}')
            target = target[part]
        if parts[-1] not in target and not (len(parts)==1 and parts[0] in {'global_train_scales','local_train_scales','local_crop_cache'}):
            raise ValueError(f'unknown configuration key: {key}')
        try:value = json.loads(raw)
        except json.JSONDecodeError:value = raw
        old = target.get(parts[-1])
        if old is not None and not (isinstance(old,float) and isinstance(value,(int,float))):
            if type(old) is not type(value):
                raise ValueError(f'wrong type for {key}')
        target[parts[-1]] = value
    for key in PATH_KEYS:
        if config.get(key):config[key] = str(project_path(config[key]))
    config['data' if document['task']=='ball' else 'dataset'] = str(project_path(dataset['path']))
    for key in ('localizer_type', 'global_localizer_type', 'local_localizer_type'):
        if key in config and config[key] not in ('residual_fpn', 'high_resolution_se'):
            raise ValueError(f'unsupported {key}')
    if config.get('tracker_type','independent') not in ('independent','association'):
        raise ValueError('unsupported tracker_type')
    if config.get('tracker_type') == 'association':
        if document['task']!='event':raise ValueError('association uses event comparison or candidate audit configs; ball trainer remains independent')
        from ttnet.tracking.association import BeamTracker
        BeamTracker(config.get('association'))
    if document['task']=='event':
        if config.get('table_frontend'):
            from ttnet.scene.visual_cache import validate_frontend
            validate_frontend(config['table_frontend'])
            if config.get('table_checkpoint') or config.get('table_source','predicted')!='predicted':
                raise ValueError('spatial table frontend cannot use legacy/oracle table input')
            config['table_frontend']['checkpoint'] = str(project_path(config['table_frontend']['checkpoint']))
            if config['table_frontend']['source']=='shared_global' and config['table_frontend']['input']!=config['global_input']:
                raise ValueError('shared table input must match Global')
        if config.get('table_fusion'):
            spec = config['table_fusion']
            if set(spec)!={'mode','channels'} or spec['mode'] not in ('none','geometry','concat','zero','gated') or type(spec['channels']) is not int or spec['channels']<1:
                raise ValueError('invalid table_fusion configuration')
            if config.get('event_head_type')!='tcn' or not config.get('context_head') or config.get('mode')!='frozen':
                raise ValueError('table fusion currently requires frozen context TCN')
            if spec['mode']!='none' and not config.get('table_frontend'):
                raise ValueError('table geometry/visual fusion requires a trained frontend')
            if config.get('table_frontend') and spec['channels']!=config['table_frontend']['channels']:
                raise ValueError('table head/frontend channel mismatch')
            if config.get('table_geometry_version')!='stable_v1' or config.get('table_smoothing'):
                raise ValueError('first spatial table fusion requires stable_v1 geometry without smoothing')
        if config.get('event_head_type') not in ('tcn','transformer'):
            raise ValueError('unsupported event_head_type')
        if config.get('event_head_type') == 'transformer':
            from ttnet.models.event import TransformerEventHead
            TransformerEventHead(**config.get('transformer',{}))
        if config.get('tracker_type') == 'association' and config.get('mode') != 'frozen':
            raise ValueError('association experiments require frozen localization')
        if config.get('mode') not in ('frozen','joint') or config.get('window') != 13 or config.get('temporal_mode') != 'fixed3':
            raise ValueError('current events require frozen/joint mode, 13 event frames and fixed3 localization')
    for key in ('feature_batch','validation_batch'):
        if key in config and (type(config[key]) is not int or config[key]<1):raise ValueError(f'{key} must be a positive integer')
    if config['micro_batch'] < 1 or config['effective_batch']%config['micro_batch'] or config['epochs'] < 1:
        raise ValueError('positive batches/epochs and integral gradient accumulation required')
    if config.get('optimizer')!='AdamW' or config.get('lr_schedule')!='constant' or config.get('warmup_epochs',0)!=0:
        raise ValueError('current trainers implement AdamW, constant learning rate and no warmup')
    if config.get('pretrained') is not None:
        raise ValueError('locator pretrained initialization is not implemented; event initialization uses explicit global/local checkpoints')
    if config.get('local_crop_cache') is not None and not isinstance(config['local_crop_cache'],bool):
        raise ValueError('local_crop_cache must be boolean')
    augmentation=config.get('augmentation')
    if isinstance(augmentation,dict):
        if not 0<=augmentation['horizontal_flip']<=1 or augmentation['gt_crop_jitter_fraction']<0:
            raise ValueError('invalid flip probability or crop jitter')
        for key in ('brightness_scale','brightness_offset'):
            value=augmentation[key]
            if len(value)!=2 or value[0]>value[1]:raise ValueError(f'invalid {key}')
    from ttnet.temporal import temporal_mode
    from ttnet.training.scales import input_size,training_scales
    temporal_mode(config)
    for stage in ('global','local'):
        input_size(config,stage);training_scales(config,stage)
    provenance = dict(legacy=False,name=document['name'],task=document['task'],dataset=dataset,
                      sources=sources,overrides=list(overrides),config_hash=identity(config))
    return config,provenance
