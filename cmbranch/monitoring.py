"""CPU-only occupancy latches and evidence/non-evidence ADD diagnostics."""
from copy import deepcopy
from .core import storage_chars

def initial_phase():
    return {'phase':1,'sessions_seen':0,'transition_after_session':None,
            'first95_any_turn':[None]*4,'first95_end_session':[None]*4}

def advance_phase(previous, session, streams, cfg):
    state = deepcopy(previous)
    budget, threshold = cfg['environment']['bank_budget_chars'],cfg['branch']['occupancy_threshold']
    ratios = [storage_chars(s['end_bank'])/budget for s in streams]
    ordinal = state['sessions_seen']
    for stream in streams:
        g = stream['stream']
        for d in stream['decisions']:
            if d['after_chars']/budget >= threshold and state['first95_any_turn'][g] is None:
                state['first95_any_turn'][g] = {'session':session['session_id'],'ordinal':ordinal,'turn_index':d['turn_index']}
        if ratios[g] >= threshold and state['first95_end_session'][g] is None:
            state['first95_end_session'][g] = {'session':session['session_id'],'ordinal':ordinal}
    if cfg['mode'] != 'original' and state['phase'] == 1 and all(r >= threshold for r in ratios):
        state['phase'] = 2
        state['transition_after_session'] = {'session':session['session_id'],'ordinal':ordinal,
                                             'end_turn':session['end'],'occupancy':ratios}
    state['sessions_seen'] += 1
    reached = [x['ordinal'] for x in state['first95_end_session'] if x is not None]
    audit = {'phase_used':previous['phase'],'next_phase':state['phase'],'end_occupancy':ratios,
             'all_four_at_threshold_now':all(r>=threshold for r in ratios),
             'first95_end_session_gap':max(reached)-min(reached) if len(reached)==4 else None,
             'first95_any_turn':state['first95_any_turn'],'first95_end_session':state['first95_end_session']}
    return state,audit

def summarize(streams,cfg):
    rows = [d for s in streams for d in s['decisions']]
    threshold = cfg['branch']['occupancy_threshold']*cfg['environment']['bank_budget_chars']
    early = [d for d in rows if d['before_chars'] < threshold]
    def rate(evidence):
        selected = [d for d in early if d['evidence_turn']==evidence]
        adds = sum(d['executed_action']=='ADD' for d in selected)
        return {'turns':len(selected),'adds':adds,'rate':adds/len(selected) if selected else None}
    groups = [g for s in streams for g in s['branch_groups']]
    executed = [g['rows'][0] for g in groups]
    return {'decisions':len(rows),'generated_outputs':sum(d['generated_outputs'] for d in rows),
            'training_outputs':sum(d['training_outputs'] for d in rows),'branch_groups':len(groups),
            'signal_groups':sum(g['has_signal'] for g in groups),
            'invalid_executed':sum(not d['valid'] for d in rows),
            'below95_evidence_add':rate(True),'below95_non_evidence_add':rate(False),
            'scored_executed_replace_groups':sum(r['valid'] and r['action']['action']=='REPLACE' for r in executed),
            'scored_executed_replace_lost_complete':sum(r['lost_complete'] for r in executed
                if r['valid'] and r['action']['action']=='REPLACE'),
            'replace_loss_scope':'only executed REPLACE in randomly sampled training branch groups'}
