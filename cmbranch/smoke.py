"""Supervised near-capacity GPU plumbing check; no scientific training result."""
import argparse
import gc
import hashlib
import os
from .config import load, resolve
from .core import atomic, log
from .branching import messages

def adapter_hashes(backend):
    from .backend import parameter_owner
    out={k:hashlib.sha256() for k in ('manager','manager_ref')}
    for name,p in sorted(backend.model.named_parameters()):
        owner=parameter_owner(name)
        if owner:out[owner].update(p.detach().float().cpu().numpy().tobytes())
    return {k:v.hexdigest() for k,v in out.items()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cfg=load(a.config)
    from .preflight import inspect
    report,sessions,_=inspect(cfg)
    if not os.environ.get('COVERMEM_STAGE_OWNER'):raise RuntimeError('GPU smoke must be supervised')
    if cfg['execution']['require_slurm'] and not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    import torch
    from .backend import Backend
    if torch.cuda.device_count()!=1:raise RuntimeError('Expose exactly one GPU')
    output=resolve(cfg['paths']['output'])/'smoke';output.mkdir(parents=True,exist_ok=True)
    (output/'SMOKE_COMPLETE.json').unlink(missing_ok=True)
    torch.manual_seed(cfg['training']['seed']);torch.cuda.manual_seed(cfg['training']['seed'])
    backend=Backend(cfg);before=adapter_hashes(backend)
    bank,used=[],0
    for turn in sessions[0]['records']:
        if used+len(turn['content'])>43500:break
        bank.append({'memory_id':'m_'+turn['turn_id'],'content':turn['content'],'source_turn_ids':[turn['turn_id']]})
        used+=len(turn['content'])
    else:raise ValueError('Smoke fixture did not fill the bank')
    # Sample both prompt modes. Synthetic signs deliberately test backward wiring.
    rows=[backend.sample('manager',messages(bank,turn,43500,phase),42+phase,
                         cfg['rollout']['manager_max_new_tokens']) for phase in (1,2)]
    for i,row in enumerate(rows):row.update(advantage=1. if i==0 else -1.,loss_mask=[1]*len(row['response_ids']))
    stats=backend.update('manager',[rows]);after=adapter_hashes(backend)
    if {k for k in before if before[k]!=after[k]}!={'manager'}:raise RuntimeError('Adapter isolation failed')
    backend.save(output/'diagnostic_checkpoint');peak=torch.cuda.max_memory_allocated()
    del backend;gc.collect();torch.cuda.empty_cache()
    restored=Backend(cfg,output/'diagnostic_checkpoint');restored.restore_training(output/'diagnostic_checkpoint')
    if adapter_hashes(restored)!=after:raise RuntimeError('Checkpoint round-trip failed')
    atomic(output/'SMOKE_COMPLETE.json',{'status':'complete','fingerprint':report['fingerprint'],
        'scope':'two prompt modes with synthetic advantages; not K-way scientific training',
        'near_capacity_chars':used,'response_tokens':[len(r['response_ids']) for r in rows],
        'device':torch.cuda.get_device_name(0),'cuda_build':torch.version.cuda,'peak_allocated_bytes':peak,
        'adapter_hashes_before':before,'adapter_hashes_after':after,'updates':stats})
    log('gpu_smoke_complete',peak_allocated_bytes=peak)

if __name__=='__main__':main()
