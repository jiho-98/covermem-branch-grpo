from pathlib import Path
import tempfile
import unittest
import torch
from fixtures import config, data, FakeBackend, tiny_backend
from cmbranch.backend import parameter_owner
from cmbranch.train import train_loop
from cmbranch.evaluate import run_history
from cmbranch import checkpoint
from cmbranch.core import read

def real_scripted(cfg):
    backend=tiny_backend(cfg);fake=FakeBackend()
    def sample(adapter,messages,seed,max_new_tokens):
        row=fake.sample(adapter,messages,seed,max_new_tokens)
        backend.activate(adapter)
        with torch.no_grad():row['old_log_probs']=backend.log_probs(row).tolist()
        return row
    backend.sample=sample
    return backend

class ResumeTests(unittest.TestCase):
    def test_real_manager_optimizer_phase_and_rng_resume_exactly(self):
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        cfg=config();cfg['branch']['probability']=1.;sessions,pools=data()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);full=real_scripted(cfg)
            before={n:p.detach().clone() for n,p in full.model.named_parameters()}
            expected=train_loop(cfg,{'fingerprint':'tiny'},sessions,pools,full,root/'full')
            self.assertGreater(expected['updates']['manager'],0)
            self.assertEqual({parameter_owner(n) for n,p in full.model.named_parameters() if not torch.equal(before[n],p)},{'manager'})
            partial=real_scripted(cfg)
            train_loop(cfg,{'fingerprint':'tiny'},sessions,pools,partial,root/'resume',stop_after_batches=2)
            path,state=checkpoint.latest(root/'resume','tiny')
            restored=real_scripted(cfg)
            set_peft_model_state_dict(restored.model,load_file(str(path/'adapters/manager/adapter_model.safetensors')),adapter_name='manager')
            restored.restore_training(path);checkpoint.restore_python_rng(state)
            actual=train_loop(cfg,{'fingerprint':'tiny'},sessions,pools,restored,root/'resume',state)
            for key in ('banks','phases','updates','session_cursors'):self.assertEqual(actual[key],expected[key])
            for a,b in zip(full.model.parameters(),restored.model.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
            # Deterministic coin and sample identity survive a collection restart.
            left=read(root/'full/batches/batch_000003/traces.json')
            right=read(root/'resume/batches/batch_000003/traces.json')
            self.assertEqual(left,right)

    def test_evaluation_greedy_single_path_transition_and_resume(self):
        sessions,_=data();cfg=config()
        history={'source_id':'eval','records':sessions[0]['records'],
                 'questions':[{'question_id':'q','eligible':True,'evidence_ids':['t000005']}]}
        backend=FakeBackend()
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)
            result=run_history(history,backend,cfg,directory,'eval-id')
            self.assertEqual(len(backend.calls),6)
            self.assertEqual(result['state']['phase'],2)
            self.assertEqual(result['state']['transition_after_session']['end_turn'],2)
            self.assertEqual([read(p)['phase'] for p in sorted((directory/'turns').glob('*.json'))],[1,1,2,2,2,2])
            class MustNotSample:
                def sample(self,*a,**k):raise AssertionError('Completed history must not generate again')
            resumed=run_history(history,MustNotSample(),cfg,directory,'eval-id')
            self.assertEqual(result,resumed)
            self.assertIn('lost_complete_events',result['replacement_audit'])

    def test_checkpoint_prune_archives_and_corruption_guard(self):
        cfg=config();sessions,pools=data()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            train_loop(cfg,{'fingerprint':'fixture'},sessions,pools,FakeBackend(),root,stop_after_batches=1)
            import shutil
            shutil.rmtree(root/'batches/batch_000001')
            saved=checkpoint.latest(root,'fixture')
            train_loop(cfg,{'fingerprint':'fixture'},sessions,pools,FakeBackend(),root,saved[1])
            self.assertTrue((root/'batches/batch_000001/traces.json').is_file())
            self.assertEqual(len(list((root/'checkpoints').glob('batch_*'))),2)
            (checkpoint.latest(root,'fixture')[0]/'fixture.json').write_text('corrupt')
            with self.assertRaises(ValueError):checkpoint.latest(root,'fixture')
