"""All-belief signed Bellman-residual brackets for discounted optimal value."""
from fractions import Fraction as F
from .core import compile_model,belief_values,pack,solve


def _f(x):return F(*x)
def _value(alpha,x):return x*alpha[0]+(1-x)*alpha[1]
def _components(points,knots,target):
    raw=[(x,x) for x in points if knots[x]==target]
    raw.extend((lo,hi) for lo,hi in zip(points,points[1:]) if knots[lo]==knots[hi]==target)
    out=[]
    for lo,hi in sorted(raw):
        if out and lo<=out[-1][1]:out[-1][1]=max(hi,out[-1][1])
        else:out.append([lo,hi])
    return [[pack(x) for x in a] for a in out]


def certify_discounted(model,belief,*,horizon=2,candidate_limit=4096):
    """Bracket V* using min/max(T V_H - V_H)/(1-gamma), for all beliefs."""
    g,_,den,_=compile_model(model);b=belief_values(belief,den)
    if g==1:raise ValueError('discounted certificate requires discount <1')
    if type(horizon) is not int or not 0<=horizon<=3:raise ValueError('certificate horizon must be integer in 0..3; one additional backup is required')
    # A single planner call reuses its layers. Any lookahead refusal returns no certificate.
    plan=solve(model,belief,horizon=horizon+1,candidate_limit=candidate_limit)
    base=plan['layers'][horizon]['nodes'];next_nodes=plan['layers'][horizon+1]['nodes']
    points=sorted({_f(x) for n in base+next_nodes for x in n['belief0_interval']})
    segments=[];knots={}
    for lo,hi in zip(points,points[1:]):
        mid=(lo+hi)/2
        a=next(n for n in base if _f(n['belief0_interval'][0])<mid<_f(n['belief0_interval'][1]))
        c=next(n for n in next_nodes if _f(n['belief0_interval'][0])<mid<_f(n['belief0_interval'][1]))
        aa=list(map(_f,a['alpha']));ca=list(map(_f,c['alpha']));delta=[ca[s]-aa[s] for s in range(2)]
        av=[_value(aa,x) for x in [lo,hi]];cv=[_value(ca,x) for x in [lo,hi]];rv=[y-x for x,y in zip(av,cv)]
        knots[lo]=rv[0];knots[hi]=rv[1]
        segments.append({'belief0_interval':[pack(lo),pack(hi)],'base_node_id':a['id'],'lookahead_node_id':c['id'],
                         'residual_alpha':list(map(pack,delta)),'base_endpoint_values':list(map(pack,av)),
                         'lookahead_endpoint_values':list(map(pack,cv)),'residual_endpoint_values':list(map(pack,rv))})
    minimum=min(knots.values());maximum=max(knots.values());norm=max(abs(minimum),abs(maximum))
    lower=minimum/(1-g);upper=maximum/(1-g)
    value=max(_value(list(map(_f,n['alpha'])),b[0]) for n in base)
    for s in segments:
        s['infinite_value_lower_endpoints']=[pack(_f(v)+lower) for v in s['base_endpoint_values']]
        s['infinite_value_upper_endpoints']=[pack(_f(v)+upper) for v in s['base_endpoint_values']]
    return {'format':'belieflab-discounted-certificate-v1','belief':plan['belief'],'horizon':horizon,
            'lookahead_solution':plan,'base_envelope':base,'segments':segments,
            'minimum_residual':pack(minimum),'maximum_residual':pack(maximum),'residual_sup_norm':pack(norm),
            'minimum_residual_belief0_intervals':_components(points,knots,minimum),
            'maximum_residual_belief0_intervals':_components(points,knots,maximum),
            'signed_value_error_interval':[pack(lower),pack(upper)],'absolute_value_error_bound':pack(norm/(1-g)),
            'value_at_input_belief':pack(value),'residual_at_input_belief':pack(_f(plan['value'])-value),
            'infinite_value_interval_at_input_belief':[pack(value+lower),pack(value+upper)],
            'scope':'discount <1; exact extrema of T V_H - V_H over EVERY belief, with T the supplied-model optimal belief Bellman operator; signed shifts min/(1-discount), max/(1-discount) bracket infinite-horizon optimal VALUE at all beliefs; raw residual is NOT the value error; bracket does not evaluate an infinite-horizon policy, certify action optimality or bound finite-tree policy loss; lookahead tree still ends after H+1 decisions; inherited finite candidate/LCM/JSON caps do not bound rational bit cost/time/RAM; no learned-model validity or real-world safety guarantee'}
