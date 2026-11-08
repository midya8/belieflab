"""Exact fixed-sequence baseline and whole-belief feedback-advantage audit."""
from fractions import Fraction as F
from itertools import product
from .core import compile_model,belief_values,pack,_envelope,solve


def solve_open_loop(model,belief,*,horizon=2):
    """Optimize every action sequence fixed before seeing any observation."""
    g,actions,den,canonical=compile_model(model);b=belief_values(belief,den)
    if type(horizon) is not int or not 0<=horizon<=4:raise ValueError('horizon must be integer in 0..4')
    candidates=[]
    # At most 3**4=81 complete sequences. No weak, hand-picked baseline.
    for sequence in product(range(len(actions)),repeat=horizon):
        alpha=[F(0),F(0)]
        for a in reversed(sequence):
            r,t,_=actions[a]
            alpha=[r[s]+g*sum(t[s][j]*alpha[j] for j in range(2)) for s in range(2)]
        candidates.append({'alpha':alpha,'sequence':list(sequence),'sequence_id':len(candidates)})
    envelope,counts=_envelope(candidates)
    values=[sum(b[s]*c['alpha'][s] for s in range(2)) for c in candidates];best=max(values)
    ties=[i for i,v in enumerate(values) if v==best]
    nodes=[{'id':i,'sequence_id':c['sequence_id'],'sequence':c['sequence'][:],
            'alpha':list(map(pack,c['alpha'])),'belief0_interval':list(map(pack,c['interval'])),
            'value_at_interval_endpoints':[pack(x*c['alpha'][0]+(1-x)*c['alpha'][1]) for x in c['interval']]}
           for i,c in enumerate(envelope)]
    return {'format':'belieflab-open-loop-v1','model':canonical,'belief':list(map(pack,b)),'horizon':horizon,
            'sequences':[{'id':i,'sequence':c['sequence'][:],'alpha':list(map(pack,c['alpha'])),
                          'value_at_input_belief':pack(values[i])} for i,c in enumerate(candidates)],
            'envelope':nodes,'counts':counts,'value':pack(best),'optimal_sequence_ids':ties,
            'selected_sequence_id':ties[0],'selected_sequence':candidates[ties[0]]['sequence'][:],
            'scope':'all fixed action sequences over the SAME model, discount and horizon; no observation-dependent action or state access; every sequence tie at input belief retained, envelope representatives need not retain every tied sequence; zero terminal reward; at most 81 complete sequences, not a time/RAM bound'}


def _value(alpha,x):return x*alpha[0]+(1-x)*alpha[1]
def _unpack(x):return F(*x)


def compare_feedback(model,belief,*,horizon=2,candidate_limit=4096):
    """Compare optimized contingent decisions with optimized fixed sequences."""
    contingent=solve(model,belief,horizon=horizon,candidate_limit=candidate_limit)
    blind=solve_open_loop(model,belief,horizon=horizon)
    cn=contingent['layers'][-1]['nodes'];on=blind['envelope']
    points=sorted({_unpack(x) for n in cn+on for x in n['belief0_interval']})
    segments=[];knots={};area=F(0)
    for lo,hi in zip(points,points[1:]):
        mid=(lo+hi)/2
        c=next(n for n in cn if _unpack(n['belief0_interval'][0])<mid<_unpack(n['belief0_interval'][1]))
        o=next(n for n in on if _unpack(n['belief0_interval'][0])<mid<_unpack(n['belief0_interval'][1]))
        ca=list(map(_unpack,c['alpha']));oa=list(map(_unpack,o['alpha']));delta=[ca[s]-oa[s] for s in range(2)]
        cv=[_value(ca,x) for x in [lo,hi]];ov=[_value(oa,x) for x in [lo,hi]];gap=[a-b for a,b in zip(cv,ov)]
        # Every fixed sequence is feasible as a contingent tree that ignores observations.
        if min(gap)<0:raise ArithmeticError('feedback dominance invariant failed')
        area+=(hi-lo)*sum(gap)/2;knots[lo]=gap[0];knots[hi]=gap[1]
        segments.append({'belief0_interval':[pack(lo),pack(hi)],'contingent_node_id':c['id'],
                         'open_loop_node_id':o['id'],'advantage_alpha':list(map(pack,delta)),
                         'contingent_endpoint_values':list(map(pack,cv)),
                         'open_loop_endpoint_values':list(map(pack,ov)),
                         'advantage_endpoint_values':list(map(pack,gap))})
    maximum=max(knots.values());components=[]
    raw=[(x,x) for x,v in knots.items() if v==maximum]
    raw.extend((lo,hi) for lo,hi in zip(points,points[1:]) if knots[lo]==knots[hi]==maximum)
    for lo,hi in sorted(raw):
        if components and lo<=components[-1][1]:components[-1][1]=max(components[-1][1],hi)
        else:components.append([lo,hi])
    return {'format':'belieflab-feedback-comparison-v1','horizon':horizon,'belief':contingent['belief'],
            'contingent_solution':contingent,'open_loop_solution':blind,'segments':segments,
            'advantage_at_input_belief':pack(_unpack(contingent['value'])-_unpack(blind['value'])),
            'maximum_advantage':pack(maximum),'maximum_advantage_belief0_intervals':[[pack(x) for x in interval] for interval in components],
            'uniform_belief_integral':pack(area),'dominates_every_belief':True,
            'scope':'optimal observation-contingent value minus optimal fixed-sequence value over the SAME supplied model, discount, horizon and rewards; piecewise-affine gap audited on the union of both exact envelope breakpoints; nonnegative by feasible-policy inclusion, not necessarily convex; uniform integral is geometric area over belief0 in [0,1], NOT expected gain under an empirical belief distribution; maximum intervals include isolated points; observes post-transition emissions, not hidden state; not a causal sensor intervention, learned information value or real-world performance guarantee'}
