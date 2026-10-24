from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import torch
from fixtures import config, tiny_backend, row
from cmbranch.backend import parameter_owner, ppo_terms

class BackendTests(unittest.TestCase):
    def test_sample_logprobs_are_actual_policy_and_no_cuda(self):
        b=tiny_backend(config())
        sample=b.sample('manager',[{'role':'user','content':'fixture'}],23,6)
        b.activate('manager')
        with torch.no_grad(): lp=b.log_probs(sample)
        torch.testing.assert_close(lp,torch.tensor(sample['old_log_probs']),atol=1e-5,rtol=1e-5)
        self.assertEqual(sample,b.sample('manager',[{'role':'user','content':'fixture'}],23,6))
        self.assertFalse(torch.cuda.is_initialized())

    def test_only_manager_changes_and_reference_is_frozen(self):
        b=tiny_backend(config())
        self.assertEqual(set(b.model.peft_config),{'manager','manager_ref'})
        before={n:p.detach().clone() for n,p in b.model.named_parameters()}
        b.update('manager',[[row(b,'manager','full action and target')]])
        changed={parameter_owner(n) for n,p in b.model.named_parameters() if not torch.equal(before[n],p)}
        self.assertEqual(changed,{'manager'})
        with self.assertRaises(ValueError):b.update('manager_ref',[])

    def test_zero_signal_and_stale_sample_guard(self):
        b=tiny_backend(config())
        r=row(b,'manager','abc',0.)
        self.assertEqual(b.update('manager',[[r]])['optimizer_steps'],0)
        r['advantage']=1.;r['old_log_probs']=[v+5 for v in r['old_log_probs']]
        before=[p.detach().clone() for p in b.parameters['manager']]
        with self.assertRaises(RuntimeError):b.update('manager',[[r]])
        for p,q in zip(before,b.parameters['manager']):self.assertTrue(torch.equal(p,q))

    def test_loss_mask_has_zero_direct_gradient(self):
        lp=torch.tensor([-1.,-1.,-1.],requires_grad=True)
        terms,_,_=ppo_terms(lp,lp.detach(),lp.detach(),2.,.2,.02)
        (terms*torch.tensor([0.,1.,0.])).sum().backward()
        torch.testing.assert_close(lp.grad,torch.tensor([0.,-2.,0.]))

    def test_save_restore_optimizer_and_adapter_layout(self):
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        b=tiny_backend(config());b.update('manager',[[row(b,'manager','abc')]])
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);b.save(root)
            other=tiny_backend(config())
            for a in ('manager',):
                set_peft_model_state_dict(other.model,load_file(str(root/'adapters'/a/'adapter_model.safetensors')),adapter_name=a)
            other.restore_training(root)
            for (n,p),(n2,p2) in zip(b.model.named_parameters(),other.model.named_parameters()):
                self.assertEqual(n,n2);torch.testing.assert_close(p,p2,rtol=0,atol=0)
            r1=row(b,'manager','def');r2=row(other,'manager','def')
            b.update('manager',[[r1]]);other.update('manager',[[r2]])
            for p,p2 in zip(b.parameters['manager'],other.parameters['manager']):torch.testing.assert_close(p,p2,rtol=0,atol=0)

if __name__ == '__main__':unittest.main()
