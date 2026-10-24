"""Same-state full-action branches, local mean baseline, session-level scaling."""
from copy import deepcopy
import math
import random
import statistics
from .active_policy import apply_action, build_messages, make_observation, parse_action
from .core import digest, request_seed
from .rewards import alternatives, score

PHASE2_INSTRUCTION = '''\nPHASE-TWO ACTION RESTRICTION
For this decision the available action types are ADD, SKIP and REPLACE only.
DELETE and UPDATE are unavailable. Keep the same four-key output format.
Use the whole current bank and current turn. REPLACE removes one entire memory
entry, including all source turns stored in that entry, then stores current_turn.
ADD/REPLACE must fit the content-character budget. No extra eviction or truncation.
'''

def messages(bank, turn, budget, phase, error=None):
    result = build_messages(make_observation(bank, turn, budget), error)
    if phase == 2:
        result[0]['content'] += PHASE2_INSTRUCTION
    return result

def deduplicate_questions(questions):
    """Equal weight per distinct OR-of-AND evidence expression, not per union."""
    seen, output = {}, []
    for q in questions:
        key = tuple(sorted(tuple(sorted(b)) for b in alternatives(q['sufficient_sets'])))
        if key not in seen:
            seen[key] = len(output)
            output.append({**deepcopy(q), 'question_id':'evidence_'+digest(key), 'origin_question_ids':[]})
        output[seen[key]]['origin_question_ids'].append(q['question_id'])
    return output

def should_branch(cfg, history, session, stream, turn, version):
    seed = request_seed(cfg['training']['seed'],version,history,session,stream,turn,'branch_coin')
    uniform = random.Random(seed).random()
    return uniform < cfg['branch']['probability'], uniform

def validate_sample(row):
    if not row['response_ids'] or len(row['response_ids']) != len(row['old_log_probs']):
        raise ValueError('Missing actual sampled IDs/logprobs')
    if row['sampled_adapter'] != 'manager':raise ValueError('Only manager samples are permitted')
    if any(not math.isfinite(p) or p > 1e-4 for p in row['old_log_probs']):
        raise ValueError('Invalid old policy logprob')

def interpret(row, bank, turn, budget, phase):
    action = None
    try:
        if row['finish_reason'] != 'stop':raise ValueError('Truncated manager output')
        action = parse_action(row['text'])
        if phase == 2 and action['action'] not in ('ADD','SKIP','REPLACE'):
            raise ValueError('Action type unavailable in phase two')
        after = apply_action(bank, turn, action, budget)
        return action, after, True, None
    except (ValueError,TypeError,KeyError) as exc:
        return action, deepcopy(bank), False, str(exc)

def finish_group(samples, bank, turn, questions, cfg, group_id):
    if len(samples) != cfg['branch']['samples']:raise ValueError('Incorrect branch sample count')
    if not questions:raise ValueError('Phase-two question pool cannot be empty')
    before = score(bank, questions)
    rows, banks, cache = [], [], {}
    for i, sample in enumerate(samples):
        validate_sample(sample)
        action, after, valid, error = interpret(sample, bank, turn, cfg['environment']['bank_budget_chars'], 2)
        row = {**deepcopy(sample), 'group_id':group_id,'sample_index':i,'action':action,
               'valid':valid,'error':error,'loss_mask':[1]*len(sample['response_ids']),
               'before_hash':digest(bank),'after_hash':digest(after)}
        if valid:
            key = digest(after)
            if key not in cache:
                measured = score(after,questions)
                pairs = list(zip(before['per_question'],measured['per_question']))
                changes = [a['reward']-b['reward'] for b,a in pairs]
                gained = sum(a['complete']>b['complete'] for b,a in pairs)
                lost = sum(a['complete']<b['complete'] for b,a in pairs)
                cache[key] = {'reward':measured['reward']-before['reward'],'after_score':measured['reward'],
                              'gained_complete':gained,'lost_complete':lost,'net_complete':gained-lost,
                              'utility_gain':sum(max(d,0) for d in changes)/len(questions),
                              'utility_loss':sum(max(-d,0) for d in changes)/len(questions)}
            row.update(cache[key])
        else:
            row.update(reward=cfg['branch']['invalid_reward'],after_score=None,gained_complete=None,
                       lost_complete=None,net_complete=None,utility_gain=None,utility_loss=None)
        rows.append(row); banks.append(after)
    baseline = statistics.mean(r['reward'] for r in rows)
    for row in rows:row.update(raw_advantage=row['reward']-baseline,advantage=row['reward']-baseline)
    selected = rows[0]  # Never choose by reward; invalid sample 0 leaves the bank unchanged.
    return banks[0], {'group_id':group_id,'rows':rows,'baseline':baseline,'before_score':before['reward'],
                     'questions':len(questions),'executed_sample_index':0,'executed_valid':selected['valid'],
                     'distinct_valid_banks':len(cache),'has_signal':any(abs(r['advantage'])>1e-12 for r in rows)}

def normalize_session(rows, cfg):
    values = [r['advantage'] for r in rows]
    if any(not math.isfinite(v) for v in values):raise ValueError('Nonfinite advantage')
    std = statistics.pstdev(values) if values else 0.
    mode = cfg['training']['advantage_normalization']
    denominator = std if mode != 'none' and std > cfg['training']['advantage_epsilon'] else 1.
    for row in rows:
        row['raw_advantage'] = row['advantage']
        row['advantage'] /= denominator
    return {'mode':mode,'outputs':len(rows),'population_std':std,'denominator':denominator,
            'weighting':'one value per training output; no recentering',
            'zero_std_fallback':denominator==1. and std<=cfg['training']['advantage_epsilon']}
