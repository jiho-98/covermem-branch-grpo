"""Four persistent banks; a session has exactly one phase and one credit type."""
from copy import deepcopy
from .core import digest, verify_bank, request_seed, storage_chars
from .rewards import score, alternatives
from .branching import messages, validate_sample, interpret, should_branch, finish_group

def collect_stream(session, stream, initial_bank, backend, cfg, questions, version, phase):
    bank, attempts, groups, decisions, mid_bank = deepcopy(initial_bank), [], [], [], None
    budget = cfg['environment']['bank_budget_chars']
    needs_loo = phase == 1 or cfg['mode'] != 'branch'
    evidence_ids = set().union(*(b for q in questions for b in alternatives(q['sufficient_sets'])))
    verify_bank(bank,session['records'][:session['start']],budget)
    for ti in range(session['start'],session['end']):
        turn = session['records'][ti]
        before_chars, before_hash = storage_chars(bank), digest(bank)
        key = [cfg['training']['seed'],version,session['history_id'],session['session_id'],stream,ti]
        branch, uniform = (should_branch(cfg,session['history_id'],session['session_id'],stream,ti,version)
                           if phase == 2 and cfg['mode'] == 'branch' else (False,None))
        generated = 0
        if branch:
            context = messages(bank,turn,budget,phase)
            samples = [backend.sample('manager',context,
                       request_seed(*key,0,'manager') if k == 0 else request_seed(*key,'branch',k),
                       cfg['rollout']['manager_max_new_tokens']) for k in range(cfg['branch']['samples'])]
            generated = len(samples)
            bank, group = finish_group(samples,bank,turn,questions,cfg,digest([key,before_hash,'branch']))
            group.update(stream=stream,turn_index=ti,turn_id=turn['turn_id'],policy_version=version,
                         branch_uniform=uniform,before_chars=before_chars)
            groups.append(group)
            chosen = group['rows'][0]
            action, valid = chosen['action'],chosen['valid']
        else:
            error = None
            # Phase two always samples once on an unbranched turn, including the
            # matched LOO control. No best-of-K or invalid-output retry bias.
            limit = cfg['environment']['max_action_attempts'] if phase == 1 else 1
            for attempt in range(limit):
                result = backend.sample('manager',messages(bank,turn,budget,phase,error),
                                        request_seed(*key,attempt,'manager'),cfg['rollout']['manager_max_new_tokens'])
                generated += 1; validate_sample(result)
                action, after, valid, error = interpret(result,bank,turn,budget,phase)
                attempts.append({**result,'stream':stream,'turn_index':ti,'turn_id':turn['turn_id'],
                    'attempt':attempt,'policy_version':version,'valid':valid,'error':error,'action':action,
                    'loss_mask':[1]*len(result['response_ids']),'decision_end':valid or attempt==limit-1,
                    'before_hash':digest(bank),'after_hash':digest(after),'used_for_training':needs_loo})
                bank = after
                if valid:break
        decisions.append({'stream':stream,'turn_index':ti,'turn_id':turn['turn_id'],'phase':phase,
            'before_chars':before_chars,'after_chars':storage_chars(bank),'before_hash':before_hash,'after_hash':digest(bank),
            'evidence_turn':turn['turn_id'] in evidence_ids,'branched':branch,'branch_uniform':uniform,
            'valid':valid,'executed_action':action['action'] if valid else 'INVALID_NO_OP',
            'generated_outputs':generated,'training_outputs':generated if needs_loo or branch else 0})
        verify_bank(bank,session['records'][:ti+1],budget)
        if needs_loo and ti+1 == session['mid']:mid_bank = deepcopy(bank)
    if needs_loo and mid_bank is None:raise ValueError('Missing midpoint bank')
    return {'stream':stream,'phase':phase,'initial_bank_hash':digest(initial_bank),'attempts':attempts,
            'branch_groups':groups,'decisions':decisions,'mid_bank':mid_bank,'end_bank':bank,
            'mid_score':score(mid_bank,session['mid_questions']['questions']) if needs_loo else None,
            'end_score':score(bank,session['end_questions']['questions']) if needs_loo else None}
