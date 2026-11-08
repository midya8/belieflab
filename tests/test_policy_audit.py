import copy,itertools,json,random,subprocess,sys,tempfile,unittest
from fractions import Fraction as F
from pathlib import Path
from unittest.mock import patch
from belieflab import evaluate_policy,audit_policy,solve
from belieflab.demo import sensing_model
from belieflab.policy_demo import policy_demo
from test_core import trajectory,trees
from test_open_loop import crossing_points


def f(x):return F(x) if type(x) is int else F(*x)
def pack(x):return [x.numerator,x.denominator]
def tree_at(t,history):
    for z in history:t=t['children'][z]
    return t

def full_path_account(m,b,tree,h):
    # Independent complete initial-state / state-path / observation-path enumeration.
    masses={hist:[F(0),F(0)] for d in range(h+1) for hist in itertools.product(range(2),repeat=d)}
    rewards={hist:F(0) for hist in masses}
    for states in itertools.product(range(2),repeat=h+1):
        for observations in itertools.product(range(2),repeat=h):
            prob=b[states[0]]
            for d in range(h):
                a=m['actions'][tree_at(tree,observations[:d])['action']]
                prob*=f(a['transition'][states[d]][states[d+1]])*f(a['observation'][states[d+1]][observations[d]])
            for d in range(h+1):
                prefix=observations[:d];masses[prefix][states[d]]+=prob
                if d<h:rewards[prefix]+=prob*f(m['actions'][tree_at(tree,prefix)['action']]['reward'][states[d]])
    return masses,rewards


def merge(pieces):
    result=[]
    for lo,hi in sorted(pieces):
        if result and lo<=result[-1][1]:result[-1][1]=max(hi,result[-1][1])
        else:result.append([lo,hi])
    return [[pack(x) for x in a] for a in result]


class PolicyTests(unittest.TestCase):
    def test_81_models_all2187_supplied_tree_values(self):
        count=0
        for acc,cost,g,pattern in itertools.product([F(1,2),F(3,4),F(1)],[-F(1,5),F(0),F(1,5)],[F(0),F(1,2),F(1)],range(3)):
            m=sensing_model();m['discount']=pack(g);m['actions'][2]['reward']=[pack(cost)]*2;m['actions'][2]['observation']=[[pack(acc),pack(1-acc)],[pack(1-acc),pack(acc)]]
            for a in m['actions']:a['transition']=[[[1,2],[1,2]],[[1,2],[1,2]]] if pattern==1 else ([[0,1],[1,0]] if pattern==2 else [[1,0],[0,1]])
            for tree in trees(3,2):
                r=evaluate_policy(m,[[2,5],[3,5]],tree);alpha=[trajectory(m,s,tree) for s in range(2)];self.assertEqual(r['policy_alpha'],list(map(pack,alpha)));self.assertEqual(r['value'],pack((2*alpha[0]+3*alpha[1])/5));self.assertEqual(r['value'],r['alpha_value']);count+=1
        self.assertEqual(count,2187)

    def check_paths(self,m,tree,h):
        b=[F(2,5),F(3,5)];masses,rewards=full_path_account(m,b,tree,h);r=evaluate_policy(m,list(map(pack,b)),tree);value=F(0);positive=0
        for d,level in enumerate(r['levels']):
            self.assertEqual(level['time'],d);self.assertEqual(level['discount_weight'],pack(f(m['discount'])**d));self.assertEqual(level['total_reach_probability'],[1,1]);self.assertEqual(len(level['nodes']),2**d)
            self.assertEqual(level['total_state_mass'],[pack(sum(masses[q][s] for q in masses if len(q)==d)) for s in range(2)])
            level_value=F(0)
            for n in level['nodes']:
                hist=tuple(n['observation_history']);mass=masses[hist];reach=sum(mass);reward=rewards[hist];discounted=f(m['discount'])**d*reward
                self.assertEqual(n['state_mass'],list(map(pack,mass)));self.assertEqual(n['reach_probability'],pack(reach));self.assertEqual(n['conditional_belief'],list(map(pack,[x/reach for x in mass]))) if reach else self.assertIsNone(n['conditional_belief'])
                self.assertEqual(n['reward_mass'],pack(reward));self.assertEqual(n['discounted_reward_mass'],pack(discounted));self.assertEqual(n['conditional_reward'],pack(reward/reach)) if reach else self.assertIsNone(n['conditional_reward']);level_value+=discounted
                if d==h:self.assertIsNone(n['action']);self.assertEqual(n['outcomes'],[]);continue
                positive+=bool(reach);self.assertEqual(n['action'],tree_at(tree,hist)['action']);self.assertEqual(len(n['outcomes']),2)
                for z,outcome in enumerate(n['outcomes']):
                    nextmass=masses[hist+(z,)];prob=sum(nextmass);self.assertEqual(outcome['observation'],z);self.assertEqual(outcome['state_mass'],list(map(pack,nextmass)));self.assertEqual(outcome['reach_probability'],pack(prob));self.assertEqual(outcome['conditional_probability'],pack(prob/reach)) if reach else self.assertIsNone(outcome['conditional_probability']);self.assertEqual(outcome['conditional_belief'],list(map(pack,[x/prob for x in nextmass]))) if prob else self.assertIsNone(outcome['conditional_belief'])
            self.assertEqual(level['discounted_reward_mass'],pack(level_value));value+=level_value
        self.assertEqual(r['value'],pack(value));self.assertEqual(r['positive_decision_histories'],positive);self.assertEqual(r['decision_histories'],2**h-1);self.assertEqual(r['terminal_histories'],2**h)

    def test_all27_trees_complete_path_occupancy_and_reward_oracle(self):
        m=sensing_model();m['discount']=[2,3];m['actions'][0]['transition']=[[[1,4],[3,4]],[[2,3],[1,3]]];m['actions'][1]['transition']=[[0,1],[1,0]]
        for tree in trees(3,2):self.check_paths(m,tree,2)

    def test_all27_trees_exact_regret_crossing_cells_and_components(self):
        m=sensing_model();tt=trees(3,2);alphas=[tuple(trajectory(m,s,t) for s in range(2)) for t in tt];points=crossing_points(alphas)
        for tree,a in zip(tt,alphas):
            r=audit_policy(m,[[1,2],[1,2]],tree);gap=lambda b:max(b*x[0]+(1-b)*x[1] for x in alphas)-(b*a[0]+(1-b)*a[1]);zeros=[];maxima=[];maximum=max(gap(F(0)),gap(F(1)))
            self.assertEqual(r['maximum_regret'],pack(maximum));self.assertEqual(r['regret_at_input_belief'],pack(gap(F(1,2))));self.assertEqual(r['optimal_at_input_belief'],gap(F(1,2))==0)
            for lo,hi in zip(points,points[1:]):
                for x in [lo,(lo+hi)/2,hi]:
                    n=next(n for n in r['regret_segments'] if f(n['belief0_interval'][0])<=x<=f(n['belief0_interval'][1]));self.assertEqual(x*f(n['regret_alpha'][0])+(1-x)*f(n['regret_alpha'][1]),gap(x))
                if gap(lo)==gap(hi)==0:zeros.append((lo,hi))
                else:zeros.extend((x,x) for x in [lo,hi] if gap(x)==0)
                if gap(lo)==gap(hi)==maximum:maxima.append((lo,hi))
                else:maxima.extend((x,x) for x in [lo,hi] if gap(x)==maximum)
            self.assertEqual(r['optimality_belief0_intervals'],merge(zeros));self.assertEqual(r['maximum_regret_belief0_intervals'],merge(maxima))
            for n in r['regret_segments']:
                node=r['optimal_solution']['layers'][-1]['nodes'][n['optimal_node_id']]
                self.assertEqual(n['regret_alpha'],[pack(f(x)-y) for x,y in zip(node['alpha'],a)])
                self.assertEqual(n['regret_endpoint_values'],[pack(gap(f(x))) for x in n['belief0_interval']])

    def test_demo_branch_mass_not_conditional_reward_and_optimality_range(self):
        d=policy_demo();good=d['contingent_policy'];bad=d['fixed_exploit0_policy'];self.assertEqual(good['evaluation']['value'],[7,10]);self.assertEqual(good['optimality_belief0_intervals'],[[[13,40],[27,40]]]);self.assertEqual(bad['optimality_belief0_intervals'],[[[27,40],[1,1]]])
        for n in good['evaluation']['levels'][1]['nodes']:self.assertEqual(n['reach_probability'],[1,2]);self.assertEqual(n['conditional_reward'],[4,5]);self.assertEqual(n['reward_mass'],[2,5])
        self.assertEqual(good['maximum_regret_belief0_intervals'],[[[0,1],[0,1]],[[1,1],[1,1]]]);self.assertEqual(bad['maximum_regret'],[4,1])

    def test_isolated_optimality_point_for_a_tree_pruned_from_envelope(self):
        m=sensing_model();m['actions'][0]['reward']=[1,0];m['actions'][1]['reward']=[0,1];m['actions'][2]['reward']=[[1,2],[1,2]]
        t={'action':2,'children':[None,None]};r=audit_policy(m,[[1,2],[1,2]],t)
        self.assertTrue(r['optimal_at_input_belief']);self.assertEqual(r['optimality_belief0_intervals'],[[[1,2],[1,2]]]);self.assertEqual(r['maximum_regret'],[1,2]);self.assertNotIn(2,[n['action'] for n in r['optimal_solution']['layers'][-1]['nodes']])

    def test_horizon_zero_terminal_only_everywhere_optimal(self):
        r=audit_policy(sensing_model(),[1,0],None);self.assertEqual(r['evaluation']['horizon'],0);self.assertEqual(r['evaluation']['decision_histories'],0);self.assertEqual(r['evaluation']['terminal_histories'],1);self.assertEqual(r['evaluation']['value'],[0,1]);self.assertEqual(r['optimality_belief0_intervals'],[[[0,1],[1,1]]]);self.assertEqual(r['maximum_regret_belief0_intervals'],[[[0,1],[1,1]]]);self.check_paths(sensing_model(),None,0)

    def test_horizon4_15decisions_16terminals_and_reference_value(self):
        m=sensing_model();tree=solve(m,[[1,2],[1,2]],horizon=4)['policy_tree'];r=evaluate_policy(m,[[1,2],[1,2]],tree);self.assertEqual(r['decision_histories'],15);self.assertEqual(r['terminal_histories'],16);self.assertEqual(sum(len(x['nodes']) for x in r['levels']),31);self.assertEqual(r['value'],pack(sum(trajectory(m,s,tree) for s in range(2))/2));self.check_paths(m,tree,4)

    def test_impossible_histories_are_zero_mass_and_null_not_fake(self):
        m=sensing_model()
        for a in m['actions']:a['observation']=[[1,0],[1,0]]
        tree=policy_demo()['contingent_policy']['evaluation']['policy_tree'];r=evaluate_policy(m,[[1,2],[1,2]],tree);n=r['levels'][1]['nodes'][1];self.assertEqual(n['reach_probability'],[0,1]);self.assertEqual(n['state_mass'],[[0,1],[0,1]]);self.assertIsNone(n['conditional_belief']);self.assertIsNone(n['conditional_reward'])
        for o in n['outcomes']:self.assertIsNone(o['conditional_probability']);self.assertIsNone(o['conditional_belief']);self.assertEqual(o['reach_probability'],[0,1])
        self.check_paths(m,tree,2)

    def test_transition_then_emission_and_initial_state_reward(self):
        m=sensing_model();m['actions'][2]['transition']=[[0,1],[1,0]];tree=policy_demo()['contingent_policy']['evaluation']['policy_tree'];r=evaluate_policy(m,[[3,4],[1,4]],tree);o=r['levels'][0]['nodes'][0]['outcomes'][0];self.assertEqual(o['state_mass'],[[9,40],[3,40]]);self.assertEqual(o['conditional_belief'],[[3,4],[1,4]]);self.assertEqual(o['reach_probability'],[3,10]);self.assertEqual(r['levels'][0]['discounted_reward_mass'],[-1,10]);self.check_paths(m,tree,2)

    def test_zero_discount_keeps_probability_trace_but_zero_future_value(self):
        m=sensing_model();m['discount']=0;tree=policy_demo()['contingent_policy']['evaluation']['policy_tree'];r=evaluate_policy(m,[[1,2],[1,2]],tree);self.assertEqual(r['value'],[-1,10]);self.assertEqual(r['levels'][1]['nodes'][0]['conditional_reward'],[4,5]);self.assertEqual(r['levels'][1]['discounted_reward_mass'],[0,1]);self.assertEqual(r['levels'][2]['total_reach_probability'],[1,1])

    def test_negative_rewards_discounted_mass_and_never_optimal_tree(self):
        m=sensing_model();m['discount']=[1,2]
        for a in m['actions']:a['reward']=[-2,-2]
        tree=solve(m,[1,0],horizon=4)['policy_tree'];r=audit_policy(m,[[1,2],[1,2]],tree);self.assertEqual(r['evaluation']['value'],[-15,4]);self.assertEqual(r['optimality_belief0_intervals'],[[[0,1],[1,1]]])
        t={'action':2,'children':[{'action':1,'children':[None,None]},{'action':0,'children':[None,None]}]};r=audit_policy(sensing_model(),[[1,2],[1,2]],t);self.assertEqual(r['optimality_belief0_intervals'],[])

    def test_strict_tree_schema_depth_balance_and_cycles(self):
        leaf={'action':0,'children':[None,None]};unbalanced={'action':0,'children':[None,leaf]};cycle={'action':0,'children':[]};cycle['children']=[cycle,cycle]
        deep=None
        for _ in range(5):deep={'action':0,'children':[deep,deep]}
        for t in [True,0,[],{}, {'action':True,'children':[None,None]}, {'action':3,'children':[None,None]}, {'action':0,'children':(None,None)}, {'action':0,'children':[None]}, {'action':0,'children':[None,None],'extra':0},unbalanced,deep,cycle]:
            with self.assertRaises(ValueError):evaluate_policy(sensing_model(),[1,0],t)

    def test_schema_belief_and_combined_lcm_validation(self):
        for b in [[1,1],[True,False],[[1,0],[0,1]]]:
            with self.assertRaises(ValueError):evaluate_policy(sensing_model(),b,None)
        m=sensing_model();m['discount']=[1,1009]
        with self.assertRaises(ValueError):evaluate_policy(m,[[1,1013],[1012,1013]],None)
        m=sensing_model();m['actions'][0]['observation']=[[1,1],[1,0]]
        with self.assertRaises(ValueError):evaluate_policy(m,[1,0],None)

    def test_tree_validation_before_alpha_and_capped_audit_no_partial_result(self):
        with patch('belieflab.policy_audit._policy_alpha',side_effect=AssertionError('must not evaluate')):
            with self.assertRaises(ValueError):evaluate_policy(sensing_model(),[1,0],{'action':0,'children':[None]})
        tree=policy_demo()['contingent_policy']['evaluation']['policy_tree']
        with self.assertRaisesRegex(ValueError,'preflight'):audit_policy(sensing_model(),[1,0],tree,candidate_limit=1)

    def test_input_output_alias_and_rng_isolation(self):
        m=sensing_model();b=[[1,2],[1,2]];leaf={'action':0,'children':[None,None]};t={'action':0,'children':[leaf,leaf]};before=copy.deepcopy((m,b,t));state=random.getstate();r=audit_policy(m,b,t);self.assertEqual((m,b,t),before);self.assertEqual(random.getstate(),state)
        r['evaluation']['policy_tree']['children'][0]['action']=1;self.assertEqual(r['evaluation']['policy_tree']['children'][1]['action'],0);r['evaluation']['levels'][0]['nodes'][0]['state_mass'][0][0]=99;self.assertEqual((m,b,t),before)

    def test_cli_matches_evaluate_audit_and_demo(self):
        with tempfile.TemporaryDirectory() as d:
            model=Path(d)/'m.json';policy=Path(d)/'p.json';m=sensing_model();t=policy_demo()['contingent_policy']['evaluation']['policy_tree'];model.write_text(json.dumps(m));policy.write_text(json.dumps(t))
            for command,api in [('evaluate-policy',evaluate_policy),('audit-policy',audit_policy)]:
                run=subprocess.run([sys.executable,'-m','belieflab',command,str(model),'--belief','[[1,2],[1,2]]','--policy',str(policy)],capture_output=True,text=True);self.assertEqual(run.returncode,0,run.stderr);self.assertEqual(json.loads(run.stdout),api(m,[[1,2],[1,2]],t))
            run=subprocess.run([sys.executable,'-m','belieflab','policy-demo'],capture_output=True,text=True);self.assertEqual(run.returncode,0,run.stderr);self.assertEqual(json.loads(run.stdout),policy_demo())

    def test_cli_invalid_duplicate_missing_oversize_policy(self):
        with tempfile.TemporaryDirectory() as d:
            model=Path(d)/'m.json';policy=Path(d)/'p.json';model.write_text(json.dumps(sensing_model()))
            for raw in ['{"action":0,"action":0,"children":[null,null]}',' '*65537,'{','{"action":0,"children":[null]}']:
                policy.write_text(raw);run=subprocess.run([sys.executable,'-m','belieflab','audit-policy',str(model),'--belief','[1,0]','--policy',str(policy)],capture_output=True,text=True);self.assertEqual(run.returncode,2);self.assertEqual(run.stdout,'');self.assertIn('error:',run.stderr)
            policy.unlink();run=subprocess.run([sys.executable,'-m','belieflab','evaluate-policy',str(model),'--belief','[1,0]','--policy',str(policy)],capture_output=True,text=True);self.assertEqual(run.returncode,2)
