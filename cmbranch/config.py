"""Strict experiment identity; separate outputs for each scientific condition."""
from copy import deepcopy
import math
from pathlib import Path
from .core import ROOT, read, digest, file_hash

def resolve(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()

def load(path):
    cfg = read(path)
    if cfg['schema'] != 'covermem.branch_grpo.v1' or cfg['mode'] not in ('branch', 'matched', 'original'):
        raise ValueError('Unknown experiment schema/mode')
    env, branch, train = cfg['environment'], cfg['branch'], cfg['training']
    if env != {'streams': 4, 'bank_budget_chars': 43500, 'max_action_attempts': 3}:
        raise ValueError('Production requires 4 streams / 43500 chars / 3 phase-one attempts')
    if cfg['manager_reward']['credit_assignment']['estimator'] != 'pre_action_bank_loo_continuation_gain_v1':
        raise ValueError('Phase one must use original action-aligned LOO credit')
    if (cfg['manager_reward']['complete'],cfg['manager_reward']['recall']) != (.7,.3):
        raise ValueError('Bank utility must remain 0.7 coverage + 0.3 recall')
    if (cfg['manager_reward']['mid_weight'],cfg['manager_reward']['end_weight']) != (.5,.5):
        raise ValueError('Mid/end weights must remain 0.5 / 0.5')
    if (cfg['rollout']['temperature'],cfg['rollout']['top_p'],cfg['rollout']['top_k']) != (1.,1.,0):
        raise ValueError('Use unwarped policy sampling')
    if train['ppo_epochs'] != 1 or train['session_batch_size'] != 1:
        raise ValueError('Exactly one session per update; never mix phase modes in an update')
    if train['dtype'] != 'bfloat16' or train['attention_implementation'] != 'sdpa':
        raise ValueError('Production uses BF16 and SDPA')
    if type(branch['samples']) is not int or branch['samples'] < 2:
        raise ValueError('K must be an integer >= 2')
    if not math.isfinite(branch['probability']) or not 0 < branch['probability'] <= 1:
        raise ValueError('Branch probability must be in (0,1]')
    if branch['occupancy_threshold'] != .95 or branch['transition'] != 'all_four_at_session_end_next_session_latched':
        raise ValueError('Switch only after all four end-of-session banks reach 95%, never switch back')
    if branch['question_pool'] != 'full_conversation_dedup_sufficient_sets':
        raise ValueError('Phase two requires full same-conversation QG, deduplicated by sufficient sets')
    if branch['executed_sample_index'] != 0 or branch['invalid_execution'] != 'no_op_no_retry':
        raise ValueError('Execute sample zero; invalid outputs cause a no-op, without resampling')
    if branch['phase2_actions'] != ['ADD','SKIP','REPLACE']:
        raise ValueError('Phase two allows ADD/SKIP/REPLACE only')
    if not math.isfinite(branch['invalid_reward']) or branch['invalid_reward'] >= -1:
        raise ValueError('Invalid penalty must be strictly below every valid bank delta')
    expected = 'none' if cfg['mode'] == 'original' else 'session_output_population_std'
    if train['advantage_normalization'] != expected or train['zero_std_rule'] != 'divide_by_one':
        raise ValueError('Unexpected session normalization/zero-variance rule')
    for name in ('manager_learning_rate','grad_clip','advantage_epsilon'):
        if not math.isfinite(train[name]) or train[name] <= 0:
            raise ValueError('Invalid ' + name)
    if train['manager_kl_coef'] != .02 or cfg['checkpoint']['keep_last'] < 2:
        raise ValueError('Retain KL=0.02 and at least two committed checkpoints')
    if 'selector_adapter' in cfg['paths'] or 'selector' in cfg:
        raise ValueError('This experiment has one trainable manager, no selector LoRA')
    return cfg

def identity(cfg, asset_report, runtime):
    science = deepcopy(cfg)
    for field in ('paths','checkpoint','execution'):science.pop(field)
    code = {p.relative_to(ROOT).as_posix(): file_hash(p) for p in sorted((ROOT/'cmbranch').glob('*.py'))}
    value = {'config': science, 'inputs': asset_report, 'code': code, 'runtime': runtime}
    return {'fingerprint':digest(value), 'identity':value}
