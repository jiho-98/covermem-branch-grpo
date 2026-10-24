"""Single-manager GRPO with a permanent per-conversation phase latch."""
import argparse
from copy import deepcopy
import fcntl
import os
from pathlib import Path
import random
import signal
import time
from .config import load, resolve
from .core import atomic, read, digest, log
from .data import make_batches
from .action_credit import attach_action_credit
from .rollout import collect_stream
from .branching import normalize_session
from .monitoring import initial_phase, advance_phase, summarize
from . import checkpoint

class Interrupted(RuntimeError):
    pass

def initial_state(sessions):
    histories = list(dict.fromkeys(s['history_id'] for s in sessions))
    return {'next_batch':0,'completed_sessions':0,'banks':{h:[[] for _ in range(4)] for h in histories},
            'session_cursors':{h:0 for h in histories},'updates':{'manager':0},
            'phases':{h:initial_phase() for h in histories}}

def train_loop(cfg, report, sessions, pools, backend, output, state=None, stop_after_batches=None):
    output = Path(output)
    state = deepcopy(state or initial_state(sessions))
    if cfg['training']['session_batch_size'] != 1:raise ValueError('One session per update is mandatory')
    schedule = make_batches(sessions,1)
    schedule_id = digest([[(s['history_id'],s['session_id'],s['start'],s['end']) for s in b] for b in schedule])
    if state.get('schedule_id',schedule_id) != schedule_id:raise ValueError('Checkpoint schedule mismatch')
    state['schedule_id'] = schedule_id
    if not 0 <= state['next_batch'] <= len(schedule):raise ValueError('Checkpoint cursor out of range')
    expected = {h:0 for h in state['session_cursors']}
    counts = {h:0 for h in expected}
    for batch in schedule[:state['next_batch']]:
        for s in batch:expected[s['history_id']]=s['end'];counts[s['history_id']]+=1
    if expected != state['session_cursors']:raise ValueError('Checkpoint history cursor mismatch')
    if any(state['phases'][h]['sessions_seen'] != n for h,n in counts.items()):
        raise ValueError('Checkpoint phase cursor mismatch')
    start_batch = state['next_batch']
    for bi in range(start_batch,len(schedule)):
        session = schedule[bi][0]
        hid = session['history_id']
        if state['session_cursors'][hid] != session['start']:raise ValueError('Chronological session mismatch')
        next_state = deepcopy(state)
        phase = state['phases'][hid]['phase']
        initial = deepcopy(state['banks'][hid])
        log('collect_session',batch=bi,history=hid,phase=phase,mode=cfg['mode'])
        start = time.perf_counter()
        streams = [collect_stream(session,i,initial[i],backend,cfg,pools[hid],bi,phase) for i in range(4)]
        if phase == 2 and cfg['mode']=='branch':
            rows = [r for stream in streams for group in stream['branch_groups'] for r in group['rows']]
            credit = {'estimator':'same_state_sample_mean_delta','mid_end_scoring':False}
        else:
            rows,credit = attach_action_credit(streams,session,initial,
                {**cfg['manager_reward'],'bank_budget_chars':cfg['environment']['bank_budget_chars']})
        normalization = normalize_session(rows,cfg)
        collection_seconds = time.perf_counter()-start
        next_state['phases'][hid],transition = advance_phase(state['phases'][hid],session,streams,cfg)
        monitor = summarize(streams,cfg)
        all_generated = [r for stream in streams for r in stream['attempts']]
        all_generated += [r for stream in streams for group in stream['branch_groups'] for r in group['rows']]
        monitor.update(generated_prompt_tokens=sum(len(r['prompt_ids']) for r in all_generated),
                       generated_response_tokens=sum(len(r['response_ids']) for r in all_generated),
                       training_response_tokens=sum(len(r['response_ids']) for r in rows))
        start = time.perf_counter()
        stats = backend.update('manager',[rows])
        update_seconds = time.perf_counter()-start
        next_state['banks'][hid] = [s['end_bank'] for s in streams]
        next_state['session_cursors'][hid] = session['end']
        next_state['completed_sessions'] += 1
        next_state['updates']['manager'] += stats['optimizer_steps']
        next_state['next_batch'] = bi+1
        metrics = {'batch':bi,'history':hid,'session':session['session_id'],'mode':cfg['mode'],'phase':phase,
                   'credit':credit,'normalization':normalization,'transition':transition,'monitor':monitor,
                   'collection_seconds':collection_seconds,'update_seconds':update_seconds,'manager':stats}
        traces = [{'history_id':hid,'session_id':session['session_id'],'phase':phase,
                   'streams':streams,'training_rows':rows}]
        dest = checkpoint.commit(output,report['fingerprint'],next_state,backend,metrics,traces,cfg['checkpoint']['keep_last'])
        state = next_state
        atomic(output/'progress.json',{'status':'running','next_batch':bi+1,'total_batches':len(schedule),
               'history':hid,'phase_used':phase,'next_phase':state['phases'][hid]['phase'],
               'updates':state['updates'],'checkpoint':str(dest)})
        log('session_committed',batch=bi,phase=phase,next_phase=state['phases'][hid]['phase'],
            updates=state['updates'],branch_groups=monitor['branch_groups'])
        if stop_after_batches is not None and bi+1-start_batch >= stop_after_batches:
            atomic(output/'PAUSED.json',{'status':'paused','fingerprint':report['fingerprint'],'checkpoint':str(dest)})
            return state
    saved = checkpoint.latest(output,report['fingerprint'])
    if saved is None:raise ValueError('Cannot complete without committed checkpoint')
    atomic(output/'TRAIN_COMPLETE.json',{'status':'complete','fingerprint':report['fingerprint'],
           'checkpoint':str(saved[0].relative_to(output)),'completed_sessions':state['completed_sessions'],
           'updates':state['updates'],'phases':state['phases'],'qa_calls':0,'qg_calls':0})
    (output/'PAUSED.json').unlink(missing_ok=True)
    return state

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--train', action='store_true')
    p.add_argument('--stop-after-batches', type=int)
    a = p.parse_args()
    from .preflight import inspect
    cfg = load(a.config)
    report, sessions, pools = inspect(cfg, assets_only=not a.train)
    if not a.train:
        print({'status': report['status'], 'gpu_started': False})
        return
    if a.stop_after_batches is not None and a.stop_after_batches < 1:
        p.error('--stop-after-batches must be positive')
    if not os.environ.get('COVERMEM_STAGE_OWNER'):
        raise RuntimeError('Run via scripts/run.sh / supervisor; do not bypass allocation ownership')
    if cfg['execution']['require_slurm'] and not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('This config requires an existing Slurm allocation')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise RuntimeError('Portable backend supports exactly one process/GPU; do not torchrun it')
    import torch
    from .backend import Backend
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one GPU; multi-GPU topology is not implemented')
    output = resolve(cfg['paths']['output'])
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = read(output / 'prepared.json') if (output / 'prepared.json').exists() else report
        if old['fingerprint'] != report['fingerprint']:
            raise ValueError('Changed science/runtime/code: use a new output directory')
        atomic(output / 'prepared.json', report)
        saved = checkpoint.latest(output, report['fingerprint'])
        random.seed(cfg['training']['seed'])
        torch.manual_seed(cfg['training']['seed'])
        torch.cuda.manual_seed(cfg['training']['seed'])
        torch.set_num_threads(cfg['execution']['cpu_threads'])
        backend = Backend(cfg, saved[0] if saved else None)
        state = saved[1] if saved else None
        if saved:
            backend.restore_training(saved[0])
            checkpoint.restore_python_rng(state)
        def interrupt(*_):
            raise Interrupted('Stage stopped; resume last committed batch (uncommitted work is replayed)')
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
            signal.signal(signum, interrupt)
        try:
            train_loop(cfg, report, sessions, pools, backend, output, state, a.stop_after_batches)
        except Interrupted as exc:
            atomic(output / 'INTERRUPTED.json', {'status': 'interrupted', 'message': str(exc), 'fingerprint': report['fingerprint']})
            raise

if __name__ == '__main__':
    main()
