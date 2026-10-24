"""One greedy manager path; no branch sampling/reward-based execution at test time."""
import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
from .config import load, resolve
from .core import atomic, read, digest, verify_bank, retained_ids, storage_chars, log
from .branching import messages, interpret
from .checkpoint import latest

def run_history(history, backend, cfg, directory, fingerprint):
    directory.mkdir(parents=True,exist_ok=True)
    path = directory/'state.json'
    state = read(path) if path.exists() else {'fingerprint':fingerprint,'next_turn':0,'bank':[],
        'phase':1,'transition_after_session':None,'actions':{},'invalid_manager_attempts':0,'invalid_turns':0}
    if state['fingerprint'] != fingerprint:raise ValueError('Changed evaluation identity')
    bank, counts = state['bank'],Counter(state['actions'])
    budget = cfg['environment']['bank_budget_chars']
    verify_bank(bank,history['records'][:state['next_turn']],budget)
    records = history['records']
    for i in range(state['next_turn'],len(records)):
        turn,error,valid,attempts = records[i],None,False,[]
        phase = state['phase']
        before_ids = sorted(retained_ids(bank))
        for attempt in range(6 if phase==1 else 1):
            row = backend.sample('manager',messages(bank,turn,budget,phase,error),42,
                                 cfg['rollout']['manager_max_new_tokens'],greedy=True)
            action,after,valid,error = interpret(row,bank,turn,budget,phase)
            attempts.append({'text':row['text'],'valid':valid,'action':action,'error':error})
            if not valid:
                state['invalid_manager_attempts'] += 1
                continue
            bank = after;counts[action['action']] += 1
            break
        if not valid:state['invalid_turns'] += 1
        verify_bank(bank,records[:i+1],budget)
        session_end = i+1==len(records) or records[i+1]['session_id']!=turn['session_id']
        if cfg['mode']!='original' and phase==1 and session_end and storage_chars(bank)/budget>=cfg['branch']['occupancy_threshold']:
            state['phase']=2
            state['transition_after_session']={'session':turn['session_id'],'end_turn':i+1}
        replacement = valid and action['action']=='REPLACE'
        atomic(directory/'turns'/f'{i:06d}.json',{'index':i,'fingerprint':fingerprint,'phase':phase,
            'attempts':attempts,'after_hash':digest(bank),
            'replace_before_ids':before_ids if replacement else None,
            'replace_after_ids':sorted(retained_ids(bank)) if replacement else None})
        state.update(next_turn=i+1,bank=bank,actions=dict(counts))
        atomic(path,state)
        if i%25==0:log('eval_progress',history=history['source_id'],turn=i+1,phase=phase)
    # Official labels are inspected only AFTER every decision in the conversation.
    questions = [q for q in history['questions'] if q['eligible']]
    kept = retained_ids(bank)
    details = []
    for q in questions:
        evidence = set(q['evidence_ids']);overlap=kept&evidence
        details.append({'question_id':q['question_id'],'complete':float(overlap==evidence),
                        'recall':len(overlap)/len(evidence),'hit':float(bool(overlap))})
    lost,gained,replacements = 0,0,0
    for p in sorted((directory/'turns').glob('*.json')):
        row=read(p)
        if row['replace_before_ids'] is None:continue
        replacements+=1
        before,after=set(row['replace_before_ids']),set(row['replace_after_ids'])
        for q in questions:
            evidence=set(q['evidence_ids']);b=evidence<=before;a=evidence<=after
            lost+=b and not a;gained+=a and not b
    result={'status':'complete','fingerprint':fingerprint,'history':history['source_id'],'questions':len(details),
            'metrics':{k:sum(q[k] for q in details)/len(details) for k in ('complete','recall','hit')},
            'replacement_audit':{'replacements':replacements,'lost_complete_events':lost,'gained_complete_events':gained,
                                 'scope':'official-question evidence transitions; repeated losses count repeatedly'},
            'state':state,'per_question':details}
    atomic(directory/'result.json',result)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cfg=load(a.config)
    from .preflight import inspect
    report,_,_=inspect(cfg)
    if not os.environ.get('COVERMEM_STAGE_OWNER'):raise RuntimeError('Evaluation requires supervised execution')
    if cfg['execution']['require_slurm'] and not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    import torch
    from .backend import Backend
    if torch.cuda.device_count()!=1:raise RuntimeError('Expose exactly one GPU')
    output=resolve(cfg['paths']['output']);complete=read(output/'TRAIN_COMPLETE.json')
    if complete['status']!='complete' or complete['fingerprint']!=report['fingerprint']:
        raise ValueError('Evaluation requires validated completed training')
    saved=latest(output,report['fingerprint'])
    if saved is None or saved[0].resolve()!=(output/complete['checkpoint']).resolve():
        raise ValueError('Final checkpoint mismatch')
    histories=read(resolve(cfg['paths']['assets'])/'evaluation_histories.json')
    if len(histories)!=5 or sum(q['eligible'] for h in histories for q in h['questions'])!=729:
        raise ValueError('Expected held-out five / 729 questions')
    protocol='greedy_own_session_end_phase_6_then_1_attempts_v1' if cfg['mode']!='original' else 'greedy_original_6_attempts_v1'
    eval_id=digest({'train':report['fingerprint'],'protocol':protocol})
    evaluation=output/'evaluation';evaluation.mkdir(parents=True,exist_ok=True)
    with (evaluation/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        backend=Backend(cfg,saved[0])
        results=[run_history(h,backend,cfg,evaluation/h['source_id'],eval_id) for h in histories]
        total=sum(r['questions'] for r in results)
        atomic(evaluation/'EVAL_COMPLETE.json',{'status':'complete','fingerprint':report['fingerprint'],
            'evaluation_fingerprint':eval_id,'protocol':protocol,'questions':total,
            'pooled':{k:sum(r['metrics'][k]*r['questions'] for r in results)/total for k in ('complete','recall','hit')},
            'replacement_audit':{k:sum(r['replacement_audit'][k] for r in results)
                for k in ('replacements','lost_complete_events','gained_complete_events')},
            'histories':[{k:v for k,v in r.items() if k not in ('state','per_question')} for r in results],
            'qa_reader_calls':0,'scope':'RAW evidence preservation, not QA accuracy'})

if __name__=='__main__':main()
