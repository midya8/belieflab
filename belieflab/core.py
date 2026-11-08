"""Exact two-state/two-observation finite-horizon POMDP policy envelopes."""
from fractions import Fraction as F
from itertools import product
from math import lcm


def keys(d,names):
    if type(d) is not dict or set(d)!=set(names):raise ValueError('object keys must be '+','.join(names))
def rational(x):
    if type(x) is int:x=[x,1]
    if type(x) is not list or len(x)!=2 or any(type(v) is not int for v in x) or abs(x[0])>1000000 or not 1<=x[1]<=1000000:raise ValueError('use bounded integer or [numerator,positive denominator], not floats/bools')
    return F(*x)
def pack(x):return [x.numerator,x.denominator]
def row(x):
    if type(x) is not list or len(x)!=2:raise ValueError('two entries required')
    return list(map(rational,x))
def simplex(x):
    r=row(x)
    if any(v<0 for v in r) or sum(r)!=1:raise ValueError('row must sum exactly to one with nonnegative entries')
    return r

def compile_model(model):
    keys(model,['version','discount','actions'])
    if type(model['version']) is not int or model['version']!=1:raise ValueError('version must be 1')
    g=rational(model['discount'])
    if not 0<=g<=1:raise ValueError('finite-horizon discount must be in [0,1]')
    actions=model['actions']
    if type(actions) is not list or not 1<=len(actions)<=3:raise ValueError('1..3 actions required')
    compiled=[];den=g.denominator
    for a in actions:
        keys(a,['reward','transition','observation']);r=row(a['reward'])
        if any(abs(v)>100 for v in r):raise ValueError('absolute rewards must be <=100')
        matrices=[]
        for key in ['transition','observation']:
            if type(a[key]) is not list or len(a[key])!=2:raise ValueError('2x2 matrices required')
            matrices.append([simplex(x) for x in a[key]])
        t,o=matrices
        for v in r+[v for m in matrices for x in m for v in x]:den=lcm(den,v.denominator)
        compiled.append((r,t,o))
    if den>1000000:raise ValueError('model denominator LCM exceeds 1000000')
    canonical={'version':1,'discount':pack(g),'actions':[{'reward':list(map(pack,r)),'transition':[list(map(pack,x)) for x in t],'observation':[list(map(pack,x)) for x in o]} for r,t,o in compiled]}
    return g,compiled,den,canonical

def belief_values(b,den):
    b=simplex(b)
    for x in b:den=lcm(den,x.denominator)
    if den>1000000:raise ValueError('combined model/belief denominator LCM exceeds 1000000')
    return b


def belief_update(model,belief,action,observation):
    g,actions,den,canonical=compile_model(model);b=belief_values(belief,den)
    if type(action) is not int or not 0<=action<len(actions) or type(observation) is not int or observation not in [0,1]:raise ValueError('invalid action or observation index')
    r,t,o=actions[action];predicted=[sum(b[s]*t[s][j] for s in range(2)) for j in range(2)];weights=[predicted[j]*o[j][observation] for j in range(2)];prob=sum(weights)
    return {'format':'belieflab-update-v1','action':action,'observation':observation,'belief':list(map(pack,b)),
            'predicted':list(map(pack,predicted)),'unnormalized':list(map(pack,weights)),
            'observation_probability':pack(prob),'posterior':list(map(pack,[x/prob for x in weights])) if prob else None,
            'scope':'transition occurs first, then observation is conditioned on new state and selected action; a zero-probability observation has posterior=null, not a fabricated belief'}


def _envelope(candidates):
    # alpha(b)=alpha[1]+b*(alpha[0]-alpha[1]), b=P(state0).
    # Equal vectors retain the first generated representative policy only.
    unique={}
    for c in candidates:unique.setdefault(tuple(c['alpha']),c)
    lines={}
    for c in unique.values():
        x,y=c['alpha'];slope=x-y
        if slope not in lines or y>lines[slope]['alpha'][1]:lines[slope]=c
    hull=[];starts=[]
    for slope,c in sorted(lines.items()):
        while hull:
            old=hull[-1];old_slope=old['alpha'][0]-old['alpha'][1]
            cross=(old['alpha'][1]-c['alpha'][1])/(slope-old_slope)
            if len(hull)==1 or cross>starts[-1]:break
            hull.pop();starts.pop()
        starts.append(cross if hull else None);hull.append(c)
    kept=[]
    for i,c in enumerate(hull):
        lo=max(F(0),starts[i]) if starts[i] is not None else F(0)
        hi=min(F(1),starts[i+1]) if i+1<len(hull) else F(1)
        if lo<hi:kept.append({**c,'interval':[lo,hi]})
    stats={'generated_candidates':len(candidates),'duplicate_vectors_removed':len(candidates)-len(unique),
           'parallel_dominated_removed':len(unique)-len(lines),'hull_removed':len(lines)-len(hull),
           'outside_belief_domain_removed':len(hull)-len(kept),'retained_vectors':len(kept)}
    assert sum(stats[k] for k in stats if k!='generated_candidates')==len(candidates)
    assert kept and kept[0]['interval'][0]==0 and kept[-1]['interval'][1]==1
    return kept,stats


def _preflight(action_count,previous_count,total,limit,depth):
    planned=action_count*previous_count**2
    if planned>limit or total+planned>8192:raise ValueError('candidate preflight refused horizon '+str(depth)+': '+str(planned)+' candidates; per-layer limit '+str(limit)+', aggregate limit 8192; no partial plan returned')
    return planned


def solve(model,belief,*,horizon=2,candidate_limit=4096):
    g,actions,den,canonical=compile_model(model);b=belief_values(belief,den)
    if type(horizon) is not int or not 0<=horizon<=4:raise ValueError('horizon must be integer in 0..4')
    if type(candidate_limit) is not int or not 1<=candidate_limit<=4096:raise ValueError('candidate_limit must be integer in 1..4096')
    previous=[{'alpha':[F(0),F(0)],'action':None,'children':[],'interval':[F(0),F(1)]}];raw_layers=[previous];layers=[];total=0;root_values=[]
    def packed(nodes):
        return [{'id':i,'alpha':list(map(pack,c['alpha'])),'action':c['action'],'children':c['children'][:],
                 'belief0_interval':list(map(pack,c['interval'])),
                 'value_at_interval_endpoints':[pack(p*c['alpha'][0]+(1-p)*c['alpha'][1]) for p in c['interval']]}
                for i,c in enumerate(nodes)]
    layers.append({'horizon':0,'nodes':packed(previous),'counts':{'generated_candidates':1,'duplicate_vectors_removed':0,'parallel_dominated_removed':0,'hull_removed':0,'outside_belief_domain_removed':0,'retained_vectors':1}})
    for depth in range(1,horizon+1):
        planned=_preflight(len(actions),len(previous),total,candidate_limit,depth)
        candidates=[]
        for a,(r,t,o) in enumerate(actions):
            for children in product(range(len(previous)),repeat=2):
                alpha=[r[s]+g*sum(t[s][j]*sum(o[j][z]*previous[children[z]]['alpha'][j] for z in range(2)) for j in range(2)) for s in range(2)]
                candidates.append({'alpha':alpha,'action':a,'children':list(children)})
        root_values=[max(sum(b[s]*c['alpha'][s] for s in range(2)) for c in candidates if c['action']==a) for a in range(len(actions))]
        previous,counts=_envelope(candidates);total+=planned;raw_layers.append(previous);layers.append({'horizon':depth,'nodes':packed(previous),'counts':counts,'root_action_values_at_input_belief':list(map(pack,root_values))})
    values=[sum(p*x for p,x in zip(b,n['alpha'])) for n in previous];best=max(values);ties=[i for i,v in enumerate(values) if v==best];chosen=ties[0]
    def tree(depth,node):
        n=raw_layers[depth][node]
        return None if depth==0 else {'action':n['action'],'children':[tree(depth-1,k) for k in n['children']]}
    return {'format':'belieflab-solution-v1','model':canonical,'belief':list(map(pack,b)),'horizon':horizon,
            'layers':layers,'candidate_limit':candidate_limit,'generated_candidates_total':total,
            'retained_node_values':list(map(pack,values)),'value':pack(best),'selected_node':chosen,
            'retained_maximizers':ties,'root_action_values':list(map(pack,root_values)),
            'optimal_root_actions':[a for a,v in enumerate(root_values) if v==best],'action':previous[chosen]['action'],'policy_tree':tree(horizon,chosen),
            'scope':'two hidden states, two post-transition observations, finite horizon <=4 with zero terminal value; exact upper envelope over every belief, not a belief grid; duplicate vectors, zero-width-only supporting policies and redundant tied plans may be removed, so returned representatives/ties are NOT all optimal policy trees; optimal_root_actions is computed separately from all root-action candidates at the input belief and DOES retain all optimal root actions; first retained maximizing representative is selected; caps limit candidate counts, not rational bit cost, bytes, time or RAM; no infinite-horizon convergence, learned model or real-world safety guarantee'}
