from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from fixtures import config, data, FakeBackend
from cmbranch.active_policy import apply_action
from cmbranch.branching import deduplicate_questions, finish_group, interpret, messages, normalize_session, should_branch
from cmbranch.monitoring import initial_phase, advance_phase, summarize
from cmbranch.rollout import collect_stream
from cmbranch.train import train_loop
from cmbranch.core import read, ROOT
from cmbranch import checkpoint

def sample(name,target=None,turn='t000002',finish='stop'):
    action={'thought':'A content-based choice.','action':name,'target_id':target,
            'source_turn_ids':[] if name in ('SKIP','DELETE') else [turn]}
    text=json.dumps(action)
    return {'text':text,'response_ids':list(text.encode()),'prompt_ids':[1,2,3],
            'old_log_probs':[-2.]*len(text),'sampled_adapter':'manager','finish_reason':finish}

def bank_fixture():
    sessions,pools=data();records=sessions[0]['records']
    bank=[{'memory_id':'m_'+r['turn_id'],'content':r['content'],'source_turn_ids':[r['turn_id']]} for r in records[:2]]
    questions=[{'question_id':'old','type':'direct_recall','sufficient_sets':['t000000'],'available_at':1},
               {'question_id':'new','type':'direct_recall','sufficient_sets':['t000002'],'available_at':999}]
    return bank,records[2],questions

class BranchTests(unittest.TestCase):
    def test_full_pool_local_mean_and_predesignated_execution(self):
        bank,turn,questions=bank_fixture();cfg=config()
        rows=[sample('REPLACE','m_t000000'),sample('REPLACE','m_t000001'),sample('REPLACE','m_t000001'),sample('SKIP')]
        after,g=finish_group(rows,bank,turn,questions,cfg,'local')
        self.assertEqual([r['reward'] for r in g['rows']],[0.,.5,.5,0.])
        self.assertEqual([r['advantage'] for r in g['rows']],[-.25,.25,.25,-.25])
        self.assertEqual(g['baseline'],.25)
        self.assertNotIn('t000000',{t for e in after for t in e['source_turn_ids']})
        self.assertEqual(g['executed_sample_index'],0)
        self.assertEqual([r['text'] for r in g['rows']],[r['text'] for r in rows])
        self.assertTrue(all(all(r['loss_mask']) for r in g['rows']))
        self.assertEqual(bank[0]['source_turn_ids'],['t000000'])

    def test_invalid_first_is_no_op_no_best_candidate_fallback(self):
        bank,turn,questions=bank_fixture()
        rows=[sample('ADD'),sample('REPLACE','m_t000001'),sample('DELETE','m_t000000'),sample('REPLACE','absent')]
        after,g=finish_group(rows,bank,turn,questions,config(),'g')
        self.assertEqual(after,bank)
        self.assertEqual([r['valid'] for r in g['rows']],[False,True,False,False])
        self.assertEqual([r['reward'] for r in g['rows']],[-2.,.5,-2.,-2.])
        self.assertTrue(g['rows'][0]['advantage']<0)
        self.assertIsNone(g['rows'][0]['after_score'])
        for name in ('UPDATE','DELETE'):
            self.assertFalse(interpret(sample(name,'m_t000000'),bank,turn,8,2)[2])

    def test_add_when_space_and_replace_capacity_violation(self):
        bank,turn,questions=bank_fixture();cfg=config()
        self.assertTrue(interpret(sample('ADD'),bank[:1],turn,8,2)[2])
        long=deepcopy(turn);long['content']='123456789'
        self.assertFalse(interpret(sample('REPLACE','m_t000001'),bank,long,8,2)[2])
        self.assertFalse(interpret(sample('SKIP',finish='length'),bank,turn,8,2)[2])

    def test_combined_entry_removed_as_whole(self):
        bank,turn,questions=bank_fixture();cfg=config();cfg['environment']['bank_budget_chars']=20
        combined=[{'memory_id':'merged','content':'AAAA\nBBBB','source_turn_ids':['t000000','t000001']}]
        rows=[sample('REPLACE','merged'),sample('SKIP'),sample('ADD'),sample('SKIP')]
        after,g=finish_group(rows,combined,turn,questions,cfg,'g')
        self.assertEqual(after[0]['source_turn_ids'],['t000002'])
        self.assertEqual(g['rows'][0]['lost_complete'],1)

    def test_dedup_preserves_or_and_distinction_and_future_questions(self):
        def q(i,sets):return {'question_id':str(i),'sufficient_sets':sets,'type':'direct_recall'}
        qs=[q(0,[['t000000','t000001'],'t000002']),q(1,['t000002',['t000001','t000000']]),
            q(2,[['t000000','t000001','t000002']])]
        got=deduplicate_questions(qs)
        self.assertEqual(len(got),2)
        self.assertEqual(got[0]['origin_question_ids'],['0','1'])
        self.assertEqual(qs[0]['question_id'],'0')
        bank,turn,questions=bank_fixture()
        rows=[sample('REPLACE','m_t000001'),sample('SKIP'),sample('SKIP'),sample('SKIP')]
        _,g=finish_group(rows,bank,turn,questions,config(),'future')
        self.assertEqual(g['rows'][0]['reward'],.5)

    def test_session_scale_does_not_recenter_or_recompute_local_baselines(self):
        cfg=config();rows=[{'advantage':x} for x in [-.25,.25,-2.,2.]]
        report=normalize_session(rows,cfg)
        self.assertEqual([r['raw_advantage'] for r in rows],[-.25,.25,-2.,2.])
        self.assertAlmostEqual(rows[1]['advantage']/rows[3]['advantage'],.125)
        self.assertNotEqual(report['denominator'],1.)
        same=[{'advantage':-2.}]*4
        # Use independent outputs, as in production.
        same=[dict(x) for x in same]
        self.assertEqual(normalize_session(same,cfg)['denominator'],1.)
        self.assertEqual([r['advantage'] for r in same],[-2.]*4)
        self.assertEqual(normalize_session([],cfg)['outputs'],0)

    def test_branch_randomness_replays_and_is_not_periodic(self):
        cfg=config();a=[should_branch(cfg,'h','s',0,i,3) for i in range(1000)]
        self.assertEqual(a,[should_branch(cfg,'h','s',0,i,3) for i in range(1000)])
        n=sum(flag for flag,_ in a);self.assertTrue(200<n<300,n)
        b=[should_branch(cfg,'h','s',1,i,3) for i in range(1000)]
        self.assertNotEqual(a,b)
        cfg['branch']['probability']=1.
        self.assertTrue(all(should_branch(cfg,'h','s',0,i,3)[0] for i in range(30)))

    def test_phase_latch_requires_simultaneous_end_occupancy_and_never_reverts(self):
        cfg=config();sessions,_=data();phase=initial_phase()
        def streams(chars):
            return [{'stream':i,'end_bank':[{'content':'x'*n}],
                     'decisions':[{'after_chars':n,'turn_index':1}]} for i,n in enumerate(chars)]
        phase,a=advance_phase(phase,sessions[0],streams([8,8,8,4]),cfg)
        self.assertEqual(phase['phase'],1)
        phase,a=advance_phase(phase,sessions[1],streams([4,8,8,8]),cfg)
        self.assertEqual(phase['phase'],1) # All have crossed historically, but not now.
        phase,a=advance_phase(phase,sessions[2],streams([8,8,8,8]),cfg)
        self.assertEqual((a['phase_used'],a['next_phase']),(1,2))
        phase,a=advance_phase(phase,sessions[2],streams([4,4,4,4]),cfg)
        self.assertEqual(phase['phase'],2)
        cfg['mode']='original'
        self.assertEqual(advance_phase(initial_phase(),sessions[0],streams([8]*4),cfg)[0]['phase'],1)

    def test_phase_two_only_selected_groups_train_and_no_mid_end_score(self):
        cfg=config();sessions,pools=data();bank,_,_=bank_fixture();backend=FakeBackend()
        with patch('cmbranch.rollout.score',side_effect=AssertionError('No midpoint/end scoring in branch phase two')):
            got=collect_stream(sessions[1],0,bank,backend,cfg,pools['fixture'],1,2)
        branched=sum(d['branched'] for d in got['decisions'])
        self.assertEqual(len(backend.calls),2+3*branched)
        self.assertEqual(sum(len(g['rows']) for g in got['branch_groups']),4*branched)
        self.assertTrue(all(not r['used_for_training'] for r in got['attempts']))
        self.assertIsNone(got['mid_score']);self.assertIsNone(got['end_score'])
        prompt=messages(bank,sessions[1]['records'][2],8,2)
        self.assertEqual(set(json.loads(prompt[1]['content'])),{'storage_budget_chars','storage_used_chars','current_active','current_turn'})

    def test_training_transition_no_mixed_credit_and_recovery(self):
        sessions,pools=data();cfg=config();cfg['branch']['probability']=1.
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);backend=FakeBackend()
            full=train_loop(cfg,{'fingerprint':'fixture'},sessions,pools,backend,root/'full')
            self.assertEqual(full['phases']['fixture']['transition_after_session']['ordinal'],0)
            metrics=[read(p) for p in sorted((root/'full/batches').glob('*/metrics.json'))]
            self.assertEqual([m['phase'] for m in metrics],[1,2,2])
            self.assertEqual(metrics[0]['credit']['estimator'],'pre_action_bank_loo_continuation_gain_v1')
            self.assertEqual(metrics[1]['credit']['estimator'],'same_state_sample_mean_delta')
            self.assertEqual(metrics[1]['monitor']['training_outputs'],32)
            train_loop(cfg,{'fingerprint':'fixture'},sessions,pools,FakeBackend(),root/'resumed',stop_after_batches=1)
            saved=checkpoint.latest(root/'resumed','fixture')
            actual=train_loop(cfg,{'fingerprint':'fixture'},sessions,pools,FakeBackend(),root/'resumed',saved[1])
            for key in ('banks','phases','session_cursors','updates','next_batch'):self.assertEqual(actual[key],full[key])
            with self.assertRaises(ValueError):checkpoint.latest(root/'resumed','changed')

    def test_matched_control_restricts_actions_but_keeps_LOO_in_phase_two(self):
        cfg=config('matched');sessions,pools=data();backend=FakeBackend()
        with tempfile.TemporaryDirectory() as temp:
            train_loop(cfg,{'fingerprint':'matched'},sessions,pools,backend,temp)
            m=read(Path(temp)/'batches/batch_000002/metrics.json')
            self.assertEqual(m['phase'],2)
            self.assertEqual(m['credit']['estimator'],'pre_action_bank_loo_continuation_gain_v1')
            self.assertEqual(m['monitor']['branch_groups'],0)
            self.assertEqual(m['monitor']['generated_outputs'],8)
            self.assertEqual(m['normalization']['mode'],'session_output_population_std')

    def test_session_with_no_randomly_selected_states_has_no_LOO_or_update(self):
        cfg=config();sessions,pools=data();backend=FakeBackend()
        with tempfile.TemporaryDirectory() as temp, patch('cmbranch.rollout.should_branch',return_value=(False,.99)):
            train_loop(cfg,{'fingerprint':'no-branches'},sessions,pools,backend,temp)
            for i in (2,3):
                m=read(Path(temp)/f'batches/batch_{i:06d}/metrics.json')
                self.assertEqual(m['credit']['estimator'],'same_state_sample_mean_delta')
                self.assertEqual(m['normalization']['outputs'],0)
                self.assertEqual(m['manager']['optimizer_steps'],0)
                self.assertEqual(m['monitor']['training_outputs'],0)

    def test_analysis_aggregates_counts_and_marks_incomplete_evaluation(self):
        from cmbranch.analyze import analyze
        cfg=config();sessions,pools=data()
        with tempfile.TemporaryDirectory() as temp:
            train_loop(cfg,{'fingerprint':'analysis'},sessions,pools,FakeBackend(),temp)
            report=analyze(temp)
            self.assertEqual(report['status'],'training_complete')
            self.assertIsNone(report['evaluation'])
            self.assertEqual(report['phases']['1']['sessions'],1)
            self.assertEqual(report['phases']['2']['sessions'],2)
            self.assertEqual(report['first95_end_session_gaps']['fixture'],0)

    def test_new_conversation_starts_again_in_phase_one(self):
        cfg=config();sessions,pools=data();other=deepcopy(sessions)
        for s in other:s['history_id']='second'
        pools['second']=deepcopy(pools['fixture'])
        with tempfile.TemporaryDirectory() as temp:
            end=train_loop(cfg,{'fingerprint':'two-histories'},sessions+other,pools,FakeBackend(),temp)
            self.assertEqual(read(Path(temp)/'batches/batch_000004/metrics.json')['phase'],1)
            self.assertEqual(end['phases']['fixture']['phase'],2)
            self.assertEqual(end['phases']['second']['transition_after_session']['ordinal'],0)

    def test_add_monitor_separates_evidence_and_non_evidence(self):
        cfg=config();streams=[{'decisions':[
            {'before_chars':0,'evidence_turn':True,'executed_action':'ADD','valid':True,'generated_outputs':1,'training_outputs':1},
            {'before_chars':4,'evidence_turn':True,'executed_action':'SKIP','valid':True,'generated_outputs':1,'training_outputs':1},
            {'before_chars':4,'evidence_turn':False,'executed_action':'ADD','valid':True,'generated_outputs':1,'training_outputs':1},
            {'before_chars':8,'evidence_turn':True,'executed_action':'SKIP','valid':True,'generated_outputs':1,'training_outputs':1}],
            'branch_groups':[]}]
        stats=summarize(streams,cfg)
        self.assertEqual(stats['below95_evidence_add']['rate'],.5)
        self.assertEqual(stats['below95_non_evidence_add']['rate'],1.)

class ConfigTests(unittest.TestCase):
    def test_all_configs_and_invalid_mixed_update(self):
        from cmbranch.config import load
        for path in sorted((ROOT/'configs').glob('*.json')):
            cfg=load(path)
            self.assertEqual(cfg['branch']['samples'],4)
            self.assertNotIn('selector_adapter',cfg['paths'])
        for key,value in [('samples',1),('probability',0),('executed_sample_index',1)]:
            cfg=read(ROOT/'configs/branch_p025.json');cfg['branch'][key]=value
            with tempfile.TemporaryDirectory() as temp:
                path=Path(temp)/'bad.json';path.write_text(json.dumps(cfg))
                with self.assertRaises(ValueError):load(path)
