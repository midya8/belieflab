import copy,itertools,json,subprocess,sys,tempfile,unittest
from fractions import Fraction as F
from pathlib import Path
from unittest.mock import patch
from belieflab import solve,belief_update
from belieflab.core import _envelope
from belieflab.demo import demo,sensing_model


def f(x):return F(x) if type(x) is int else F(*x)
def pack(x):return [x.numerator,x.denominator]
def trees(actions,h):
    if not h:return [None]
    prev=trees(actions,h-1)
    return [{'action':a,'children':[x,y]} for a in range(actions) for x in prev for y in prev]
def trajectory(model,state,tree):
    if tree is None:return F(0)
    a=model['actions'][tree['action']];v=f(a['reward'][state])
    for nextstate in range(2):
        for obs in range(2):
            probability=f(a['transition'][state][nextstate])*f(a['observation'][nextstate][obs])
            v+=f(model['discount'])*probability*trajectory(model,nextstate,tree['children'][obs])
    return v

def belief_recursion(model,b,h):
    if not h:return F(0)
    values=[]
    for a in model['actions']:
        reward=sum(b[s]*f(a['reward'][s]) for s in range(2));future=F(0)
        for obs in range(2):
            mass=[sum(b[s]*f(a['transition'][s][j]) for s in range(2))*f(a['observation'][j][obs]) for j in range(2)];prob=sum(mass)
            if prob:future+=prob*belief_recursion(model,[x/prob for x in mass],h-1)
        values.append(reward+f(model['discount'])*future)
    return max(values)

def envelope_check(test,alphas,nodes):
    points={F(0),F(1)}
    for a,b in itertools.combinations(set(alphas),2):
        slope=a[0]-a[1]-b[0]+b[1]
        if slope:
            x=(b[1]-a[1])/slope
            if 0<=x<=1:points.add(x)
    ordered=sorted(points);points.update((a+b)/2 for a,b in zip(ordered,ordered[1:]))
    for x in points:
        expected=max(a[0]*x+a[1]*(1-x) for a in alphas)
        actual=max(f(n['alpha'][0])*x+f(n['alpha'][1])*(1-x) for n in nodes)
        test.assertEqual(actual,expected)


class CoreTests(unittest.TestCase):
    def test_81_models_all_27_depth2_trees_and_all_crossing_cells(self):
        count=0
        for acc,cost,discount,pattern in itertools.product([F(1,2),F(3,4),F(1)],[-F(1,5),F(0),F(1,5)],[F(0),F(1,2),F(1)],range(3)):
            m=sensing_model();m['discount']=pack(discount);m['actions'][2]['reward']=[pack(cost),pack(cost)];m['actions'][2]['observation']=[[pack(acc),pack(1-acc)],[pack(1-acc),pack(acc)]]
            for a in m['actions']:
                a['transition']=[[[1,2],[1,2]],[[1,2],[1,2]]] if pattern==1 else ([[0,1],[1,0]] if pattern==2 else [[1,0],[0,1]])
            r=solve(m,[[1,2],[1,2]],horizon=2);alltrees=trees(3,2);count+=len(alltrees);alphas=[tuple(trajectory(m,s,t) for s in range(2)) for t in alltrees]
            root_values=[max(sum(alpha,F(0))/2 for alpha,tree in zip(alphas,alltrees) if tree['action']==a) for a in range(3)]
            self.assertEqual(r['root_action_values'],list(map(pack,root_values)))
            self.assertEqual(r['optimal_root_actions'],[i for i,v in enumerate(root_values) if v==max(root_values)])
            envelope_check(self,alphas,r['layers'][-1]['nodes'])
            self.assertEqual(f(r['value']),belief_recursion(m,[F(1,2)]*2,2))
            self.assertEqual([pack(trajectory(m,s,r['policy_tree'])) for s in range(2)],r['layers'][-1]['nodes'][r['selected_node']]['alpha'])
            self.check_layers(m,r)
        self.assertEqual(count,2187)

    def check_layers(self,m,result):
        for layer in result['layers']:
            nodes=layer['nodes'];counts=layer['counts'];self.assertEqual(counts['generated_candidates'],sum(v for k,v in counts.items() if k!='generated_candidates'))
            self.assertEqual(nodes[0]['belief0_interval'][0],[0,1]);self.assertEqual(nodes[-1]['belief0_interval'][1],[1,1])
            for i,n in enumerate(nodes):
                self.assertEqual(n['id'],i);lo,hi=map(f,n['belief0_interval']);self.assertLess(lo,hi)
                if i:self.assertEqual(nodes[i-1]['belief0_interval'][1],n['belief0_interval'][0])
                for endpoint,value in zip([lo,hi],n['value_at_interval_endpoints']):self.assertEqual(f(value),endpoint*f(n['alpha'][0])+(1-endpoint)*f(n['alpha'][1]))
                if not layer['horizon']:continue
                prev=result['layers'][layer['horizon']-1]['nodes'];a=m['actions'][n['action']]
                for s in range(2):
                    expected=f(a['reward'][s])+f(m['discount'])*sum(f(a['transition'][s][j])*f(a['observation'][j][z])*f(prev[n['children'][z]]['alpha'][j]) for j in range(2) for z in range(2))
                    self.assertEqual(expected,f(n['alpha'][s]))

    def test_depth3_all2187_policy_trees_and_belief_recursion(self):
        m=sensing_model();alltrees=trees(3,3);self.assertEqual(len(alltrees),2187);alphas=[tuple(trajectory(m,s,t) for s in range(2)) for t in alltrees];r=solve(m,[[1,2],[1,2]],horizon=3)
        points={F(i,20) for i in range(21)}|{f(x) for n in r['layers'][-1]['nodes'] for x in n['belief0_interval']}
        for x in points:
            expected=max(x*a[0]+(1-x)*a[1] for a in alphas);actual=max(x*f(n['alpha'][0])+(1-x)*f(n['alpha'][1]) for n in r['layers'][-1]['nodes']);self.assertEqual(expected,actual);self.assertEqual(actual,belief_recursion(m,[x,1-x],3))
        self.check_layers(m,r)

    def test_sensing_demo_exact_thresholds_posterior_and_tree(self):
        d=demo();self.assertEqual(d['two_step']['value'],[7,10]);self.assertEqual(d['two_step']['action'],2);self.assertEqual([n['belief0_interval'] for n in d['two_step']['layers'][-1]['nodes']],[[[0,1],[13,40]],[[13,40],[27,40]],[[27,40],[1,1]]]);self.assertEqual(d['probe_branches'][0]['observation_probability'],[1,2])

    def test_horizon0_and1(self):
        m=sensing_model();r=solve(m,[1,0],horizon=0);self.assertIsNone(r['action']);self.assertIsNone(r['policy_tree']);self.assertEqual(r['value'],[0,1]);self.assertEqual(r['generated_candidates_total'],0)
        for x in [F(0),F(1,4),F(1,2),F(1)]:self.assertEqual(f(solve(m,[pack(x),pack(1-x)],horizon=1)['value']),abs(2*x-1))

    def test_boundary_ties_are_retained_representatives_not_all_plans(self):
        m=sensing_model();r=solve(m,[[13,40],[27,40]],horizon=2);self.assertEqual(len(r['retained_maximizers']),2);self.assertIn('NOT all optimal',r['scope'])

    def test_update_transition_before_observation(self):
        m=sensing_model();m['actions'][2]['transition']=[[0,1],[1,0]];r=belief_update(m,[[3,4],[1,4]],2,0);self.assertEqual(r['predicted'],[[1,4],[3,4]]);self.assertEqual(r['unnormalized'],[[9,40],[3,40]]);self.assertEqual(r['observation_probability'],[3,10]);self.assertEqual(r['posterior'],[[3,4],[1,4]])

    def test_zero_probability_observation_returns_null(self):
        m=sensing_model();m['actions'][0]['observation']=[[1,0],[1,0]];r=belief_update(m,[1,0],0,1);self.assertEqual(r['observation_probability'],[0,1]);self.assertIsNone(r['posterior'])

    def test_observation_probabilities_and_mixture_recover_prediction(self):
        m=sensing_model()
        for a in range(3):
            rs=[belief_update(m,[[2,3],[1,3]],a,z) for z in range(2)];self.assertEqual(sum(f(r['observation_probability']) for r in rs),1)
            for j in range(2):self.assertEqual(sum(f(r['observation_probability'])*f(r['posterior'][j]) for r in rs),f(rs[0]['predicted'][j]))

    def test_envelope_duplicate_parallel_zero_width_and_domain_pruning(self):
        candidates=[{'alpha':list(map(F,a)),'action':i,'children':[]} for i,a in enumerate([[1,-1],[-1,1],[0,0],[1,-1],[-2,-2],[3,-100]])]
        nodes,stats=_envelope(candidates);envelope_check(self,[tuple(c['alpha']) for c in candidates],[{'alpha':[pack(x) for x in n['alpha']]} for n in nodes]);self.assertEqual(stats['duplicate_vectors_removed'],1);self.assertGreater(stats['hull_removed'],0)

    def test_endpoint_only_plan_may_be_removed_but_value_preserved(self):
        cs=[{'alpha':[F(0),F(0)],'action':0,'children':[]},{'alpha':[F(-1),F(0)],'action':1,'children':[]}];nodes,stats=_envelope(cs);self.assertEqual(len(nodes),1);self.assertEqual(nodes[0]['action'],0)

    def test_per_layer_preflight_happens_before_backup(self):
        with patch('belieflab.core._envelope',side_effect=AssertionError('must not back up')):
            with self.assertRaisesRegex(ValueError,'preflight refused horizon 1'):solve(sensing_model(),[1,0],candidate_limit=2)
        with self.assertRaisesRegex(ValueError,'preflight refused horizon 2'):solve(sensing_model(),[1,0],candidate_limit=3)

    def test_horizon4_and_count_contract(self):
        r=solve(sensing_model(),[[1,2],[1,2]],horizon=4);self.check_layers(sensing_model(),r);self.assertEqual(r['generated_candidates_total'],sum(l['counts']['generated_candidates'] for l in r['layers'][1:]));self.assertTrue(all(l['counts']['generated_candidates']<=4096 for l in r['layers']))

    def test_strict_schema(self):
        for field,v in [('version',True),('version',2),('actions',[]),('actions',[sensing_model()['actions'][0]]*4),('discount',[-1,2]),('discount',2),('discount',True),('discount',.5)]:
            m=sensing_model();m[field]=v
            with self.assertRaises(ValueError):solve(m,[1,0])
        m=sensing_model();m['extra']=1
        with self.assertRaises(ValueError):solve(m,[1,0])

    def test_strict_reward_rational_and_matrices(self):
        for field,v in [('reward',[101,0]),('reward',[True,0]),('reward',[[1,0],0]),('reward',[[1000001,1],0]),('reward',['1',0]),('transition',[[1,1],[0,1]]),('observation',[[1,0],[-1,2]]),('observation',[[1,0]])]:
            m=sensing_model();m['actions'][0][field]=v
            with self.assertRaises(ValueError):solve(m,[1,0])

    def test_combined_denominator_budget(self):
        m=sensing_model();m['actions'][0]['reward']=[[1,997],0]
        with self.assertRaisesRegex(ValueError,'combined'):solve(m,[[1,991],[990,991]])
        m['actions'][1]['reward']=[[1,983],0]
        with self.assertRaisesRegex(ValueError,'model denominator'):solve(m,[1,0])

    def test_invalid_beliefs_parameters_indices(self):
        m=sensing_model()
        for b in [[1,1],[],[True,0],[[-1,2],[3,2]],[.5,.5]]:
            with self.assertRaises(ValueError):solve(m,b)
        for h in [True,-1,5,.5]:
            with self.assertRaises(ValueError):solve(m,[1,0],horizon=h)
        for c in [True,0,4097,1.5]:
            with self.assertRaises(ValueError):solve(m,[1,0],candidate_limit=c)
        for a,z in [(True,0),(3,0),(0,True),(0,2)]:
            with self.assertRaises(ValueError):belief_update(m,[1,0],a,z)

    def test_discount0_ignores_future(self):
        m=sensing_model();m['discount']=0
        for h in range(1,5):self.assertEqual(solve(m,[[1,4],[3,4]],horizon=h)['value'],[1,2])

    def test_duplicate_actions_do_not_claim_all_optimal_roots(self):
        m=sensing_model();m['actions']=[m['actions'][0],copy.deepcopy(m['actions'][0])];r=solve(m,[1,0],horizon=2);self.assertEqual(len(r['layers'][-1]['nodes']),1);self.assertEqual(r['action'],0);self.assertEqual(r['value'],[2,1]);self.assertEqual(r['optimal_root_actions'],[0,1])

    def test_independence_and_no_input_mutation(self):
        m=sensing_model();before=copy.deepcopy(m);r=solve(m,[[1,2],[1,2]]);r['model']['actions'][0]['reward'][0][0]=99;r['layers'][0]['nodes'][0]['alpha'][0][0]=99;self.assertEqual(m,before);self.assertEqual(r['layers'][1]['nodes'][0]['alpha'][0],[-1,1]);self.assertEqual(solve(m,[[1,2],[1,2]])['value'],[7,10])

    def test_cli_all_commands_and_errors(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'m.json';path.write_text(json.dumps(sensing_model()))
            for args in [['demo'],['solve',str(path),'--belief','[[1,2],[1,2]]'],['update',str(path),'--belief','[[1,2],[1,2]]','--action','2','--observation','0']]:
                r=subprocess.run([sys.executable,'-m','belieflab',*args],capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr);self.assertIsInstance(json.loads(r.stdout),dict)
            for raw in ['{"version":1,"version":1}', ' '*262145]:
                path.write_text(raw);r=subprocess.run([sys.executable,'-m','belieflab','solve',str(path),'--belief','[1,0]'],capture_output=True,text=True);self.assertEqual(r.returncode,2);self.assertEqual(r.stdout,'')

    def test_preflight_aggregate_boundary_and_per_layer_boundary(self):
        from belieflab.core import _preflight
        self.assertEqual(_preflight(3,36,4304,4096,4),3888)
        with self.assertRaises(ValueError):_preflight(3,36,4305,4096,4)
        self.assertEqual(_preflight(1,64,0,4096,1),4096)
        with self.assertRaises(ValueError):_preflight(1,65,0,4096,1)

    def test_one_action_constant_negative_rewards_all_horizons(self):
        m=sensing_model();m['actions']=[m['actions'][0]];m['actions'][0]['reward']=[-1,-1];m['discount']=[1,2]
        for h in range(5):self.assertEqual(f(solve(m,[[1,2],[1,2]],horizon=h)['value']),-sum(F(1,2)**i for i in range(h)))

    def test_envelope_rational_crossing_not_float_rounded(self):
        cs=[{'alpha':[F(1),F(0)],'action':0,'children':[]},{'alpha':[F(0),F(1,999983)],'action':1,'children':[]}];ns,stats=_envelope(cs);self.assertEqual(ns[0]['interval'][1],F(1,999984));self.assertEqual(ns[1]['interval'][0],F(1,999984))

    def test_observation_action_and_row_shape_errors(self):
        for field,value in [('transition',{}),('observation',[[[1,2],[1,2],[0,1]],[[1,2],[1,2]]]),('reward',None)]:
            m=sensing_model();m['actions'][0][field]=value
            with self.assertRaises(ValueError):solve(m,[1,0])
        m=sensing_model();m['actions'][0]['extra']=0
        with self.assertRaises(ValueError):solve(m,[1,0])

    def test_no_rng_usage(self):
        import random
        before=random.getstate();solve(sensing_model(),[[1,2],[1,2]],horizon=4);self.assertEqual(before,random.getstate())
