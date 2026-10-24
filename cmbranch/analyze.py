"""Read-only CPU aggregation: ADD rates, phase delays, cost and final evidence."""
import argparse
from pathlib import Path
from .core import atomic, read

def analyze(output):
    output=Path(output)
    paths=sorted((output/'batches').glob('batch_*/metrics.json'))
    metrics=[read(p) for p in paths]
    complete=read(output/'TRAIN_COMPLETE.json') if (output/'TRAIN_COMPLETE.json').exists() else None
    evaluation=read(output/'evaluation/EVAL_COMPLETE.json') if (output/'evaluation/EVAL_COMPLETE.json').exists() else None
    if complete and evaluation and complete['fingerprint']!=evaluation['fingerprint']:
        raise ValueError('Training/evaluation fingerprints differ')
    if complete and len(metrics)!=complete['completed_sessions']:
        raise ValueError('Completed run has missing session metrics')
    def add_rate(key):
        turns=sum(m['monitor'][key]['turns'] for m in metrics)
        adds=sum(m['monitor'][key]['adds'] for m in metrics)
        return {'turns':turns,'adds':adds,'rate':adds/turns if turns else None}
    phase_totals={}
    for phase in (1,2):
        selected=[m for m in metrics if m['phase']==phase]
        phase_totals[str(phase)]={'sessions':len(selected),
            **{key:sum(m['monitor'][key] for m in selected) for key in ('decisions','generated_outputs','training_outputs',
                'generated_prompt_tokens','generated_response_tokens','training_response_tokens','branch_groups','signal_groups','invalid_executed')},
            **{key:sum(m[key] for m in selected) for key in ('collection_seconds','update_seconds')}}
    phases=complete['phases'] if complete else {}
    report={'run':output.name,'status':'complete' if complete and evaluation else 'training_complete' if complete else 'incomplete',
            'fingerprint':complete['fingerprint'] if complete else None,'committed_sessions':len(metrics),
            'below95_evidence_add':add_rate('below95_evidence_add'),
            'below95_non_evidence_add':add_rate('below95_non_evidence_add'),'phases':phase_totals,
            'transition_records':phases,'evaluation':evaluation,
            'interpretation':'ADD rates are observational, not matched-state causal estimates. Training output expectation is not measured FLOPs/time parity.'}
    if complete:
        report['first95_end_session_gaps']={}
        for hid,phase in phases.items():
            ordinals=[x['ordinal'] for x in phase['first95_end_session'] if x is not None]
            report['first95_end_session_gaps'][hid]=max(ordinals)-min(ordinals) if len(ordinals)==4 else None
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',action='append',required=True);p.add_argument('--report',required=True)
    p.add_argument('--require-complete',action='store_true')
    a=p.parse_args();results=[analyze(x) for x in a.run]
    if a.require_complete and any(r['status']!='complete' for r in results):raise ValueError('All runs must finish training and evaluation')
    payload={'runs':results,'gpu_used':False,'status':'complete' if all(r['status']=='complete' for r in results) else 'partial'}
    if len(results)==1:payload['fingerprint']=results[0]['fingerprint']
    atomic(a.report,payload)
    for r in results:
        cov=r['evaluation']['pooled']['complete'] if r['evaluation'] else None
        print({'run':r['run'],'status':r['status'],'coverage':cov,
               'evidence_add_rate':r['below95_evidence_add']['rate'],
               'non_evidence_add_rate':r['below95_non_evidence_add']['rate'],
               'phase2_sessions':r['phases']['2']['sessions']})

if __name__=='__main__':main()
