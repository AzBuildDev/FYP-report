"""Finite instruction contract and scene/capability validation. No hardware imports."""
import json
import re
import time
from 红蓝语义配置 import LEGACY_PROFILE, task_texts, policy_binding

TASK_TEXT = 'Pick up the red cube and place it in the target area.'
TASKS = {f'{c}_cube_to_zone_{z}': {'color': c, 'object': 'cube', 'target': f'zone_{z}'}
         for c in ('red', 'blue', 'green') for z in ('A', 'B')}
CAPABILITIES = {'red_cube_to_zone_A': TASK_TEXT}
COLORS = {'red': r'\b(red|scarlet|crimson)\b|红', 'blue': r'\b(blue|azure)\b|蓝',
          'green': r'\b(green|emerald)\b|绿'}

class InstructionError(ValueError):
    pass

def strict_json(raw):
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise InstructionError('duplicate_json_key')
            result[k] = v
        return result
    try:
        spec = json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda x: (_ for _ in ()).throw(InstructionError('nonfinite_json')))
    except (ValueError, TypeError) as e:
        raise InstructionError('invalid_json') from e
    if type(spec) is not dict or set(spec) != {'status', 'task_id'}:
        raise InstructionError('invalid_schema')
    if spec['status'] not in ('parsed', 'clarify', 'unsupported'):
        raise InstructionError('invalid_status')
    if spec['status'] == 'parsed':
        if type(spec['task_id']) is not str or spec['task_id'] not in TASKS:
            raise InstructionError('unknown_task_id')
    elif spec['task_id'] is not None:
        raise InstructionError('nonparsed_task_must_be_null')
    return spec

def grounding(text, default_target=None):
    """Conservative independent veto; this never substitutes for LLM classification."""
    if not isinstance(text, str) or not text.strip():
        return {'status': 'clarify', 'task_id': None, 'reason': 'empty_transcript'}
    t = text.lower().strip()
    def result(status, reason, task=None):
        return {'status': status, 'task_id': task, 'reason': reason}
    if len(t) > 1000:
        return result('unsupported', 'input_too_long')
    if re.search(r'ignore|system prompt|developer|instruction|json|task_id|serial|joint|firmware|servo|degrees|关节|忽略|系统提示|串口', t):
        return result('unsupported', 'unsafe_or_injection')
    if re.search(r'\b(stack|throw|rotate|pour|push|all|both|two|three|outside|table edge|cup|bottle|sphere|ball|purple|orange|black|white)\b|堆|扔|旋转|所有|两个|杯|紫|桌外', t):
        return result('unsupported', 'unsupported_action_or_object')
    # Preserve the full original text for the model; only the independent veto uses positive clauses.
    clauses = re.split(r"[.!?;。；]|\bbut\b|而是|,\s*(?=(?:put|place|move|take|pick|grab|transfer|carry|bring)\b)", t)
    positives = [c for c in clauses if not re.search(r"\b(don't|dont|do not|not|never)\b|不要|别|不是", c)]
    positive = ' '.join(positives)
    colors = [c for c, pattern in COLORS.items() if re.search(pattern, positive)]
    if len(colors) != 1:
        return result('clarify', 'missing_or_multiple_positive_colors')
    if not re.search(r'\b(cube|block)\b|方块|积木', positive):
        return result('clarify', 'missing_object')
    if not re.search(r'\b(put|place|move|take|pick|grab|transfer|carry|bring|deposit)\b|放|拿|移动|抓', positive):
        return result('unsupported', 'missing_supported_action')
    zones = []
    if re.search(r'zone[ _-]*a\b|area[ _-]*a\b|yellow (area|zone)|target (area|zone)|目标区|黄色区域|a区', positive):
        zones.append('A')
    if re.search(r'zone[ _-]*b\b|area[ _-]*b\b|blue (area|zone)|b区', positive):
        zones.append('B')
    if len(zones) > 1:
        return result('clarify', 'multiple_targets')
    if not zones and default_target not in ('zone_A', 'zone_B'):
        return result('clarify', 'missing_target')
    z = zones[0] if zones else default_target[-1]
    return result('parsed', 'grounded', f'{colors[0]}_cube_to_zone_{z}')

def template_classify(text, default_target=None):
    spec = grounding(text, default_target)
    return {k: spec[k] for k in ('status', 'task_id')}

def validate_scene(scene, task_id, now=None):
    if type(scene) is not dict:
        raise InstructionError('scene_required')
    if scene.get('busy') is not False:
        raise InstructionError('robot_busy_or_unknown')
    stamp = scene.get('observed_at')
    now = time.time() if now is None else now
    if type(stamp) not in (float, int) or not 0 <= now - stamp <= 30:
        raise InstructionError('scene_stale_or_invalid')
    task = TASKS[task_id]
    objects = scene.get('objects')
    zones = scene.get('zones')
    if not isinstance(objects, list) or not isinstance(zones, list):
        raise InstructionError('invalid_scene_schema')
    matches = [o for o in objects if isinstance(o, dict) and o.get('color') == task['color'] and o.get('object') == task['object']]
    if not matches:
        raise InstructionError('object_not_seen')
    if len(matches) != 1:
        raise InstructionError('ambiguous_scene_objects')
    if task['target'] not in zones or len(zones) != len(set(zones)):
        raise InstructionError('target_not_available')
    if not isinstance(matches[0].get('id'), str) or not matches[0]['id']:
        raise InstructionError('object_id_required')
    return matches[0]['id']

def compile_request(transcript, raw, scene, default_target=None, *, policy_profile=LEGACY_PROFILE):
    base = {'transcript': transcript, 'status': 'rejected', 'task_id': None, 'task_text': None,
            'dry_run': True, 'executable': False, 'policy_profile': policy_profile}
    try:
        capabilities = task_texts(policy_profile)
        if default_target is not None and (not isinstance(scene, dict) or scene.get('zones') != [default_target]):
            raise InstructionError('default_target_requires_single_scene_zone')
        spec = strict_json(raw)
        guard = grounding(transcript, default_target)
        if guard['status'] != 'parsed':
            return {**base, 'status': guard['status'], 'reason': guard['reason']}
        if spec['status'] != 'parsed':
            return {**base, **spec, 'reason': 'model_requires_clarification' if spec['status'] == 'clarify' else 'model_unsupported'}
        if spec['task_id'] != guard['task_id']:
            raise InstructionError('model_grounding_mismatch')
        base['task_id'] = spec['task_id']
        object_id = validate_scene(scene, spec['task_id'])
        if spec['task_id'] not in capabilities:
            raise InstructionError('policy_not_validated_for_task')
        return {**base, 'status': 'ready', 'reason': 'dry_run_only', 'task_text': capabilities[spec['task_id']],
                'policy_binding': policy_binding(policy_profile),
                'object_id': object_id, 'scene_observed_at': scene['observed_at']}
    except (InstructionError, TypeError, KeyError, ValueError) as e:
        return {**base, 'reason': str(e)}

class DryRunExecutor:
    """Explicit review boundary. No hardware executor is supplied or automatically imported."""
    def __init__(self, policy_profile=LEGACY_PROFILE):
        task_texts(policy_profile)
        self.policy_profile = policy_profile

    def execute(self, request, scene):
        if request.get('status') != 'ready' or request.get('dry_run') is not True:
            raise InstructionError('request_not_ready')
        if request.get('policy_profile', LEGACY_PROFILE) != self.policy_profile:
            raise InstructionError('policy_profile_mismatch')
        capabilities = task_texts(self.policy_profile)
        if request.get('policy_binding') != policy_binding(self.policy_profile):
            raise InstructionError('policy_binding_mismatch')
        task_id = request.get('task_id')
        if task_id not in capabilities or request.get('task_text') != capabilities[task_id]:
            raise InstructionError('invalid_execution_contract')
        object_id = validate_scene(scene, task_id)
        if object_id != request.get('object_id') or scene.get('observed_at') != request.get('scene_observed_at'):
            raise InstructionError('scene_changed_replan_required')
        return {'status': 'dry_run_completed', 'task_id': task_id, 'task_text': capabilities[task_id],
                'policy_profile': self.policy_profile,
                'hardware_connected': False, 'motion_sent': False}

def fixture_scene():
    return {'source': 'synthetic_test_fixture_not_camera', 'observed_at': time.time(), 'busy': False,
            'zones': ['zone_A','zone_B'], 'objects': [
                {'id': c+'1', 'color': c, 'object': 'cube'} for c in COLORS]}
