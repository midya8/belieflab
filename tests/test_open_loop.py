import copy,itertools,json,random,subprocess,sys,tempfile,unittest
from fractions import Fraction as F
from pathlib import Path
from unittest.mock import patch
from belieflab import solve_open_loop,compare_feedback
from belieflab.demo import sensing_model
from belieflab.feedback_demo import feedback_demo
from test_core import trajectory,trees,belief_recursion


def f(x):return F(x) if type(x) is int else F(*x)
def pack(x):return [x.numerator,x.denominator]
def value(a,x):return x*a[0]+(1-x)*a[1]


def sequence_trajectory(m,start,sequence):
    # Independent full state-path sum, not the production backward recurrence.
    total=F(0)
    for path in itertools.product(range(2),repeat=len(sequence)):
        state=start;prob=F(1);reward=F(0)
        for step,(a,nextstate) in enumerate(zip(sequence,path)):
            action=m['actions'][a];reward+=f(m['discount'])**step*f(action['reward'][state])
            prob*=f(action['transition'][state][nextstate]);state=nextstate
        total+=prob*reward
    return total


def crossing_points(alphas):
    points={F(0),F(1)}
    for a,b in itertools.combinations(set(alphas),2):
        slope=a[0]-a[1]-b[0]+b[1]
        if slope:
            x=(b[1]-a[1])/slope
            if 0<x<1:points.add(x)
    return sorted(points)


class OpenLoopTests(unittest.TestCase):
    def check_envelopes(self,m,h,r):
        seqs=list(itertools.product(range(len(m['actions'])),repeat=h))
        sa=[tuple(sequence_trajectory(m,s,q) for s in range(2)) for q in seqs]
        ta=[tuple(trajectory(m,s,t) for s in range(2)) for t in trees(len(m['actions']),h)]
        blind=r['open_loop_solution'];self.assertEqual(len(blind['sequences']),len(seqs))
        for i,(q,a,n) in enumerate(zip(seqs,sa,blind['sequences'])):
            self.assertEqual(n['id'],i);self.assertEqual(n['sequence'],list(q));self.assertEqual(n['alpha'],list(map(pack,a)))
        b=f(r['belief'][0]);vals=[value(a,b) for a in sa]
        self.assertEqual(blind['value'],pack(max(vals)));self.assertEqual(blind['optimal_sequence_ids'],[i for i,v in enumerate(vals) if v==max(vals)])
        self.assertEqual(blind['selected_sequence_id'],blind['optimal_sequence_ids'][0])
        self.assertEqual(blind['selected_sequence'],list(seqs[blind['selected_sequence_id']]))
        points=crossing_points(sa+ta);area=F(0);maxgap=F(0)
        for lo,hi in zip(points,points[1:]):
            gap=[]
            for x in [lo,(lo+hi)/2,hi]:
                expected_c=max(value(a,x) for a in ta);expected_o=max(value(a,x) for a in sa);expected=expected_c-expected_o
                segment=next(n for n in r['segments'] if f(n['belief0_interval'][0])<=x<=f(n['belief0_interval'][1]))
                self.assertEqual(value(list(map(f,segment['advantage_alpha'])),x),expected)
                c=r['contingent_solution']['layers'][-1]['nodes'][segment['contingent_node_id']]
                o=blind['envelope'][segment['open_loop_node_id']]
                self.assertEqual(value(list(map(f,c['alpha'])),x),expected_c);self.assertEqual(value(list(map(f,o['alpha'])),x),expected_o)
                gap.append(expected);maxgap=max(maxgap,expected)
            area+=(hi-lo)*(gap[0]+gap[-1])/2
        self.assertEqual(r['uniform_belief_integral'],pack(area));self.assertEqual(r['maximum_advantage'],pack(maxgap))
        self.assertEqual(r['advantage_at_input_belief'],pack(max(value(a,b) for a in ta)-max(value(a,b) for a in sa)))
        gaps={x:max(value(a,x) for a in ta)-max(value(a,x) for a in sa) for x in points}
        pieces=[(x,x) for x,v in gaps.items() if v==maxgap]
        pieces.extend((lo,hi) for lo,hi in zip(points,points[1:]) if gaps[lo]==gaps[hi]==maxgap)
        expected=[]
        for lo,hi in sorted(pieces):
            if expected and lo<=expected[-1][1]:expected[-1][1]=max(hi,expected[-1][1])
            else:expected.append([lo,hi])
        self.assertEqual(r['maximum_advantage_belief0_intervals'],[[pack(x) for x in piece] for piece in expected])
        for segment in r['segments']:
            for index,x in enumerate(map(f,segment['belief0_interval'])):
                c=max(value(a,x) for a in ta);o=max(value(a,x) for a in sa)
                self.assertEqual(segment['contingent_endpoint_values'][index],pack(c));self.assertEqual(segment['open_loop_endpoint_values'][index],pack(o));self.assertEqual(segment['advantage_endpoint_values'][index],pack(c-o))
        stats=blind['counts'];self.assertEqual(stats['generated_candidates'],len(seqs));self.assertEqual(sum(v for k,v in stats.items() if k!='generated_candidates'),len(seqs))
        self.assertEqual(r['segments'][0]['belief0_interval'][0],[0,1]);self.assertEqual(r['segments'][-1]['belief0_interval'][1],[1,1])
        for a,b in zip(r['segments'],r['segments'][1:]):self.assertEqual(a['belief0_interval'][1],b['belief0_interval'][0])

    def test_81_models_all729_sequences_and2187_trees_all_crossing_cells(self):
        count=0
        for acc,reward,discount,pattern in itertools.product([F(1,2),F(3,4),F(1)],[-F(1,5),F(0),F(1,5)],[F(0),F(1,2),F(1)],range(3)):
            m=sensing_model();m['discount']=pack(discount);m['actions'][2]['reward']=[pack(reward)]*2;m['actions'][2]['observation']=[[pack(acc),pack(1-acc)],[pack(1-acc),pack(acc)]]
            for a in m['actions']:a['transition']=[[[1,2],[1,2]],[[1,2],[1,2]]] if pattern==1 else ([[0,1],[1,0]] if pattern==2 else [[1,0],[0,1]])
            r=compare_feedback(m,[[2,5],[3,5]],horizon=2);self.check_envelopes(m,2,r);count+=len(r['open_loop_solution']['sequences'])
        self.assertEqual(count,729)

    def test_analytic_sensing_gap_and_optimal_sequence_ties(self):
        r=feedback_demo()['sensing'];self.assertEqual(r['advantage_at_input_belief'],[7,10]);self.assertEqual(r['uniform_belief_integral'],[49,400]);self.assertEqual(r['maximum_advantage_belief0_intervals'],[[[1,2],[1,2]]])
        self.assertEqual(r['open_loop_solution']['optimal_sequence_ids'],[0,1,3,4]);self.assertEqual(r['open_loop_solution']['counts']['generated_candidates'],9)
        self.assertEqual([n['belief0_interval'] for n in r['segments']],[[[0,1],[13,40]],[[13,40],[1,2]],[[1,2],[27,40]],[[27,40],[1,1]]])
        self.assertEqual([n['advantage_endpoint_values'] for n in r['segments']],[[[0,1],[0,1]],[[0,1],[7,10]],[[7,10],[0,1]],[[0,1],[0,1]]])

    def test_no_information_control_zero_gap_every_belief(self):
        r=feedback_demo()['uninformative_control'];self.assertEqual(r['maximum_advantage'],[0,1]);self.assertEqual(r['maximum_advantage_belief0_intervals'],[[[0,1],[1,1]]]);self.check_envelopes(r['contingent_solution']['model'],2,r)

    def test_horizon_zero_empty_sequence_and_whole_domain_maximum(self):
        r=compare_feedback(sensing_model(),[1,0],horizon=0);self.assertEqual(r['open_loop_solution']['sequences'],[{'id':0,'sequence':[],'alpha':[[0,1],[0,1]],'value_at_input_belief':[0,1]}]);self.assertEqual(r['maximum_advantage_belief0_intervals'],[[[0,1],[1,1]]]);self.check_envelopes(sensing_model(),0,r)

    def test_one_step_never_uses_future_observation(self):
        for b in [[1,0],[0,1],[[1,2],[1,2]]]:
            r=compare_feedback(sensing_model(),b,horizon=1);self.assertEqual(r['maximum_advantage'],[0,1]);self.assertEqual(r['uniform_belief_integral'],[0,1])

    def test_maximum_interval_not_just_sampled_knots(self):
        m=sensing_model();m['actions']=m['actions'][:2]
        for a in m['actions']:a['transition']=[[[1,2],[1,2]],[[1,2],[1,2]]];a['observation']=[[1,0],[0,1]]
        r=compare_feedback(m,[[1,2],[1,2]],horizon=2);self.assertEqual(r['maximum_advantage'],[1,1]);self.assertEqual(r['uniform_belief_integral'],[1,1]);self.assertEqual(r['maximum_advantage_belief0_intervals'],[[[0,1],[1,1]]]);self.check_envelopes(m,2,r)

    def test_all81_horizon4_sequences_independent_state_paths(self):
        m=sensing_model();m['discount']=[2,3]
        for i,a in enumerate(m['actions']):a['transition']=[[[1,4],[3,4]],[[2,3],[1,3]]] if i%2 else [[0,1],[1,0]]
        r=solve_open_loop(m,[[2,5],[3,5]],horizon=4);self.assertEqual(len(r['sequences']),81)
        for n in r['sequences']:self.assertEqual(n['alpha'],[pack(sequence_trajectory(m,s,n['sequence'])) for s in range(2)])

    def test_horizon4_feedback_values_independent_belief_recursion(self):
        m=sensing_model();r=compare_feedback(m,[[1,2],[1,2]],horizon=4)
        for x in [F(0),F(1,3),F(1,2),F(1)]:
            expected=belief_recursion(m,[x,1-x],4);actual=max(value(list(map(f,n['alpha'])),x) for n in r['contingent_solution']['layers'][-1]['nodes']);self.assertEqual(actual,expected)

    def test_single_action_feedback_cannot_change_decisions(self):
        m=sensing_model();m['actions']=m['actions'][2:];r=compare_feedback(m,[[1,2],[1,2]],horizon=4);self.assertEqual(r['maximum_advantage'],[0,1]);self.assertEqual(r['open_loop_solution']['selected_sequence'],[0]*4)

    def test_zero_discount_and_negative_rewards(self):
        m=sensing_model();m['discount']=0
        for a in m['actions']:a['reward']=[-2,-3]
        r=compare_feedback(m,[[1,2],[1,2]],horizon=3);self.assertEqual(r['advantage_at_input_belief'],[0,1]);self.assertEqual(r['open_loop_solution']['value'],[-5,2]);self.assertEqual(len(r['open_loop_solution']['optimal_sequence_ids']),27)

    def test_observations_ignored_by_fixed_sequence_but_still_validated(self):
        m=sensing_model();a=solve_open_loop(m,[[1,2],[1,2]],horizon=3)
        for x in m['actions']:x['observation']=[[0,1],[0,1]]
        b=solve_open_loop(m,[[1,2],[1,2]],horizon=3);self.assertEqual(a['sequences'],b['sequences']);self.assertEqual(a['envelope'],b['envelope'])
        m['actions'][0]['observation']=[[1,1],[0,1]]
        with self.assertRaises(ValueError):solve_open_loop(m,[1,0])

    def test_invalid_horizon_schema_and_belief(self):
        for h in [-1,5,True,1.0,'2']:
            for api in [solve_open_loop,compare_feedback]:
                with self.assertRaises(ValueError):api(sensing_model(),[1,0],horizon=h)
        for b in [[1,1],[True,False],[[1,0],[0,1]],['1','0']]:
            with self.assertRaises(ValueError):solve_open_loop(sensing_model(),b)
        m=sensing_model();m['discount']=[1,1009]
        with self.assertRaises(ValueError):solve_open_loop(m,[[1,1013],[1012,1013]])

    def test_refusal_precedes_open_loop_work_no_partial_comparison(self):
        with patch('belieflab.open_loop.solve_open_loop',side_effect=AssertionError('must not run')):
            with self.assertRaisesRegex(ValueError,'preflight'):compare_feedback(sensing_model(),[1,0],horizon=2,candidate_limit=1)

    def test_input_output_and_rng_isolation(self):
        m=sensing_model();b=[[1,2],[1,2]];original=copy.deepcopy((m,b));state=random.getstate();r=compare_feedback(m,b);self.assertEqual((m,b),original);self.assertEqual(random.getstate(),state)
        r['open_loop_solution']['model']['actions'][0]['reward'][0][0]=99;r['open_loop_solution']['sequences'][0]['sequence'][0]=2
        self.assertEqual((m,b),original);self.assertEqual(compare_feedback(m,b)['open_loop_solution']['selected_sequence'],[0,0])

    def test_cli_matches_api_and_feedback_demo(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model.json';m=sensing_model();path.write_text(json.dumps(m))
            for command,api in [('open-loop',solve_open_loop),('compare',compare_feedback)]:
                run=subprocess.run([sys.executable,'-m','belieflab',command,str(path),'--belief','[[1,2],[1,2]]','--horizon','2'],capture_output=True,text=True);self.assertEqual(run.returncode,0,run.stderr);self.assertEqual(json.loads(run.stdout),api(m,[[1,2],[1,2]]))
            run=subprocess.run([sys.executable,'-m','belieflab','feedback-demo'],capture_output=True,text=True);self.assertEqual(run.returncode,0,run.stderr);self.assertEqual(json.loads(run.stdout),feedback_demo())

    def test_cli_duplicate_oversize_invalid_and_capped(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'m.json'
            for raw,options in [('{"version":1,"version":1}',[]),(' '*262145,[]),(json.dumps(sensing_model()),['--candidate-limit','1'])]:
                path.write_text(raw);run=subprocess.run([sys.executable,'-m','belieflab','compare',str(path),'--belief','[1,0]']+options,capture_output=True,text=True);self.assertEqual(run.returncode,2);self.assertEqual(run.stdout,'');self.assertIn('error:',run.stderr)
