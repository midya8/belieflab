import copy,itertools,json,random,subprocess,sys,tempfile,unittest
from fractions import Fraction as F
from pathlib import Path
from unittest.mock import patch
from belieflab import certify_discounted
from belieflab.demo import sensing_model
from belieflab.residual_demo import residual_demo
from test_core import trajectory,trees
from test_open_loop import crossing_points


def f(x):return F(x) if type(x) is int else F(*x)
def p(x):return [x.numerator,x.denominator]
def value(alphas,b):return max(b*a[0]+(1-b)*a[1] for a in alphas)
def merged(raw):
    out=[]
    for lo,hi in sorted(raw):
        if out and lo<=out[-1][1]:out[-1][1]=max(hi,out[-1][1])
        else:out.append([lo,hi])
    return [[p(x) for x in q] for q in out]


class ResidualTests(unittest.TestCase):
    def test_complete_oracle_crossing_cells_54_models(self):
        count=0
        for acc,cost,g,pattern in itertools.product([F(1,2),F(3,4),F(1)],[-F(1,5),F(0),F(1,5)],[F(0),F(1,2)],range(3)):
            m=sensing_model();m['discount']=p(g);m['actions'][2]['reward']=[p(cost)]*2;m['actions'][2]['observation']=[[p(acc),p(1-acc)],[p(1-acc),p(acc)]]
            for a in m['actions']:a['transition']=[[[1,2],[1,2]],[[1,2],[1,2]]] if pattern==1 else ([[0,1],[1,0]] if pattern==2 else [[1,0],[0,1]])
            r=certify_discounted(m,[[2,5],[3,5]],horizon=1)
            a=[[trajectory(m,s,t) for s in range(2)] for t in trees(3,1)];c=[[trajectory(m,s,t) for s in range(2)] for t in trees(3,2)]
            points=sorted(set(list(crossing_points(list(map(tuple,a))))+list(crossing_points(list(map(tuple,c))))));knots={x:value(c,x)-value(a,x) for x in points};mn=min(knots.values());mx=max(knots.values())
            self.assertEqual(r['minimum_residual'],p(mn));self.assertEqual(r['maximum_residual'],p(mx));self.assertEqual(r['residual_sup_norm'],p(max(abs(mn),abs(mx))))
            self.assertEqual(r['signed_value_error_interval'],[p(mn/(1-g)),p(mx/(1-g))])
            for target,key in [(mn,'minimum'),(mx,'maximum')]:
                raw=[(x,x) for x,v in knots.items() if v==target]+[(lo,hi) for lo,hi in zip(points,points[1:]) if knots[lo]==knots[hi]==target]
                self.assertEqual(r[key+'_residual_belief0_intervals'],merged(raw))
            for lo,hi in zip(points,points[1:]):
                for b in [lo,(lo+hi)/2,hi]:
                    seg=next(s for s in r['segments'] if f(s['belief0_interval'][0])<=b<=f(s['belief0_interval'][1]))
                    self.assertEqual(b*f(seg['residual_alpha'][0])+(1-b)*f(seg['residual_alpha'][1]),value(c,b)-value(a,b))
                    base=r['base_envelope'][seg['base_node_id']];nxt=r['lookahead_solution']['layers'][-1]['nodes'][seg['lookahead_node_id']]
                    self.assertEqual(value([list(map(f,base['alpha']))],b),value(a,b));self.assertEqual(value([list(map(f,nxt['alpha']))],b),value(c,b))
            for seg in r['segments']:
                for i,b in enumerate(map(f,seg['belief0_interval'])):
                    self.assertEqual(seg['base_endpoint_values'][i],p(value(a,b)));self.assertEqual(seg['lookahead_endpoint_values'][i],p(value(c,b)))
                    self.assertEqual(seg['infinite_value_lower_endpoints'][i],p(value(a,b)+mn/(1-g)));self.assertEqual(seg['infinite_value_upper_endpoints'][i],p(value(a,b)+mx/(1-g)))
            self.assertEqual(r['residual_at_input_belief'],p(value(c,F(2,5))-value(a,F(2,5))));count+=1
        self.assertEqual(count,54)

    def test_81_revealing_models_independent_infinite_mdp_reference(self):
        count=0
        for r0,r1,g,pattern in itertools.product([-1,0,1],[-1,0,1],[F(0),F(1,2),F(9,10)],range(3)):
            t0=[[[3,4],[1,4]],[[1,2],[1,2]]];t1=[[[1,3],[2,3]],[[1,4],[3,4]]]
            if pattern==1:t0=[[1,0],[0,1]]
            if pattern==2:t1=[[0,1],[1,0]]
            m={'version':1,'discount':p(g),'actions':[{'reward':[r0,r1],'transition':t0,'observation':[[1,0],[0,1]]},{'reward':[r1,-r0],'transition':t1,'observation':[[1,0],[0,1]]}]}
            refs=[]
            for pi in itertools.product(range(2),repeat=2):
                a0,a1=[m['actions'][pi[s]] for s in range(2)];a=1-g*f(a0['transition'][0][0]);b=-g*f(a0['transition'][0][1]);c=-g*f(a1['transition'][1][0]);d=1-g*f(a1['transition'][1][1]);x=f(a0['reward'][0]);y=f(a1['reward'][1]);det=a*d-b*c
                refs.append([(d*x-b*y)/det,(a*y-c*x)/det])
            v=[max(q[s] for q in refs) for s in range(2)]
            alphas=[[f(a['reward'][s])+g*sum(f(a['transition'][s][j])*v[j] for j in range(2)) for s in range(2)] for a in m['actions']]
            for h in range(4):
                r=certify_discounted(m,[[2,5],[3,5]],horizon=h);shift=list(map(f,r['signed_value_error_interval']))
                for b in [F(0),F(1,7),F(2,5),F(1,2),F(1)]:
                    base=value([list(map(f,n['alpha'])) for n in r['base_envelope']],b);exact=value(alphas,b)
                    self.assertLessEqual(base+shift[0],exact);self.assertLessEqual(exact,base+shift[1]);self.assertLessEqual(abs(exact-base),f(r['absolute_value_error_bound']))
            count+=1
        self.assertEqual(count,81)

    def test_perfect_probe_exact_infinite_fixed_point_and_bracket(self):
        r=residual_demo()['perfect_probe'];g=F(1,2)
        def exact(b):return max(2*abs(2*b-1),F(9,10))
        # Exact all affine cells: include certificate boundaries and true-envelope knots.
        points=sorted({F(0),F(1),F(11,40),F(29,40)}|{f(x) for s in r['segments'] for x in s['belief0_interval']})
        for lo,hi in zip(points,points[1:]):
            for b in [lo,(lo+hi)/2,hi]:
                bellman=max(2*b-1+g*exact(b),1-2*b+g*exact(b),-F(1,10)+g*2)
                self.assertEqual(exact(b),bellman)
                base=value([list(map(f,n['alpha'])) for n in r['base_envelope']],b);low,high=map(f,r['signed_value_error_interval']);self.assertLessEqual(base+low,exact(b));self.assertLessEqual(exact(b),base+high)
        self.assertEqual(r['minimum_residual'],[31,600]);self.assertEqual(r['maximum_residual'],[1,8]);self.assertEqual(r['infinite_value_interval_at_input_belief'],[[113,150],[9,10]])
        self.assertEqual(r['maximum_residual_belief0_intervals'],[[[0,1],[0,1]],[[11,35],[24,35]],[[1,1],[1,1]]])

    def test_signed_negative_and_positive_constant_exact_bounds(self):
        d=residual_demo()
        for key,sign in [('positive_constant',1),('negative_constant',-1)]:
            r=d[key];self.assertEqual(r['signed_value_error_interval'],[[sign,1],[sign,1]]);self.assertEqual(r['absolute_value_error_bound'],[1,1]);self.assertEqual(r['minimum_residual_belief0_intervals'],[[[0,1],[1,1]]]);self.assertEqual(r['maximum_residual_belief0_intervals'],[[[0,1],[1,1]]])
            self.assertEqual(r['residual_sup_norm'],[1,100])

    def test_zero_discount_horizon_one_is_exact_every_belief(self):
        m=sensing_model();m['discount']=0;r=certify_discounted(m,[[1,2],[1,2]],horizon=1)
        self.assertEqual(r['signed_value_error_interval'],[[0,1],[0,1]]);self.assertEqual(r['absolute_value_error_bound'],[0,1]);self.assertEqual(r['infinite_value_interval_at_input_belief'],[[0,1],[0,1]])

    def test_mixed_sign_residual_horizon_zero(self):
        m=sensing_model();m['actions']=m['actions'][:1];m['discount']=[1,2];r=certify_discounted(m,[[1,2],[1,2]],horizon=0)
        self.assertEqual(r['minimum_residual'],[-1,1]);self.assertEqual(r['maximum_residual'],[1,1]);self.assertEqual(r['signed_value_error_interval'],[[-2,1],[2,1]])

    def test_discount_one_and_bad_horizons_refused_before_solve(self):
        m=sensing_model()
        with patch('belieflab.residual.solve',side_effect=AssertionError('must not solve')):
            with self.assertRaisesRegex(ValueError,'discount <1'):certify_discounted(m,[[1,2],[1,2]])
            m['discount']=[1,2]
            for h in [-1,4,True,1.0,None]:
                with self.assertRaisesRegex(ValueError,'horizon'):certify_discounted(m,[[1,2],[1,2]],horizon=h)

    def test_lookahead_preflight_no_partial_certificate(self):
        m=sensing_model();m['discount']=[1,2]
        with self.assertRaisesRegex(ValueError,'preflight refused horizon 2'):certify_discounted(m,[[1,2],[1,2]],horizon=1,candidate_limit=3)
        for limit in [0,4097,True,1.5]:
            with self.assertRaises(ValueError):certify_discounted(m,[[1,2],[1,2]],candidate_limit=limit)

    def test_model_belief_validation_before_planner(self):
        m=sensing_model();m['discount']=[1,2]
        with patch('belieflab.residual.solve',side_effect=AssertionError('must not solve')):
            for b in [[True,0],[[1,3],[1,3]],[[1,1000003],[1000002,1000003]]]:
                with self.assertRaises(ValueError):certify_discounted(m,b)
            bad=copy.deepcopy(m);bad['extra']=1
            with self.assertRaises(ValueError):certify_discounted(bad,[[1,2],[1,2]])

    def test_input_output_rng_isolation_and_one_planner_call(self):
        m=sensing_model();m['discount']=[1,2];b=[[2,5],[3,5]];before=copy.deepcopy((m,b));state=random.getstate()
        from belieflab.residual import solve
        with patch('belieflab.residual.solve',wraps=solve) as spy:r=certify_discounted(m,b,horizon=2);self.assertEqual(spy.call_count,1);self.assertEqual(spy.call_args.kwargs['horizon'],3)
        expected=copy.deepcopy(r);r['base_envelope'][0]['alpha'][0][0]=999;r['lookahead_solution']['model']['actions'][0]['reward'][0][0]=999
        self.assertEqual((m,b),before);self.assertEqual(random.getstate(),state);self.assertEqual(certify_discounted(m,b,horizon=2),expected)

    def test_cli_matches_api_and_demo(self):
        with tempfile.TemporaryDirectory() as d:
            pth=Path(d)/'m.json';m=sensing_model();m['discount']=[1,2];pth.write_text(json.dumps(m))
            out=subprocess.run([sys.executable,'-m','belieflab','certify-discounted',str(pth),'--belief','[[2,5],[3,5]]','--horizon','2'],text=True,capture_output=True,check=True)
            self.assertEqual(json.loads(out.stdout),certify_discounted(m,[[2,5],[3,5]],horizon=2))
            out=subprocess.run([sys.executable,'-m','belieflab','residual-demo'],text=True,capture_output=True,check=True);self.assertEqual(json.loads(out.stdout),residual_demo())

    def test_cli_failures_no_json_output(self):
        with tempfile.TemporaryDirectory() as d:
            pth=Path(d)/'m.json'
            for raw in ['{','{"version":1,"version":1}',' '*262145,json.dumps(sensing_model())]:
                pth.write_text(raw);r=subprocess.run([sys.executable,'-m','belieflab','certify-discounted',str(pth),'--belief','[[1,2],[1,2]]'],capture_output=True,text=True);self.assertEqual(r.returncode,2);self.assertEqual(r.stdout,'');self.assertIn('error:',r.stderr)
            pth.unlink();r=subprocess.run([sys.executable,'-m','belieflab','certify-discounted',str(pth),'--belief','[[1,2],[1,2]]'],capture_output=True,text=True);self.assertEqual(r.returncode,2);self.assertEqual(r.stdout,'')

if __name__=='__main__':unittest.main()
