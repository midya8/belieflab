"""Exact forward execution and same-horizon regret of supplied policy trees."""
from fractions import Fraction as F
from .core import compile_model,belief_values,pack,keys,solve


def _validate_tree(tree,actions,depth=0):
    if tree is None:return None,0
    if depth>=4:raise ValueError('policy tree depth exceeds 4')
    keys(tree,['action','children']);a=tree['action'];children=tree['children']
    if type(a) is not int or not 0<=a<actions:raise ValueError('invalid policy action')
    if type(children) is not list or len(children)!=2:raise ValueError('policy children must be a list of two branches')
    parsed=[_validate_tree(c,actions,depth+1) for c in children]
    if parsed[0][1]!=parsed[1][1]:raise ValueError('all policy leaves must have the same horizon')
    return {'action':a,'children':[c for c,h in parsed]},1+parsed[0][1]


def _policy_alpha(tree,actions,discount):
    if tree is None:return [F(0),F(0)]
    r,t,o=actions[tree['action']];child=[_policy_alpha(c,actions,discount) for c in tree['children']]
    return [r[s]+discount*sum(t[s][j]*o[j][z]*child[z][j] for j in range(2) for z in range(2)) for s in range(2)]


def evaluate_policy(model,belief,policy_tree):
    """Sum exact unconditional history/state occupancy and reward contributions."""
    discount,actions,den,canonical=compile_model(model);b=belief_values(belief,den)
    tree,horizon=_validate_tree(policy_tree,len(actions));alpha=_policy_alpha(tree,actions,discount)
    current=[([],b,tree)];levels=[];value=F(0);positive_decisions=0
    for depth in range(horizon+1):
        records=[];following=[];level_reward=F(0)
        for history,mass,node in current:
            reach=sum(mass);a=node['action'] if node is not None else None
            reward=sum(mass[s]*actions[a][0][s] for s in range(2)) if a is not None else F(0)
            weighted=discount**depth*reward;outcomes=[];level_reward+=weighted
            if a is not None:
                positive_decisions+=bool(reach);_,t,o=actions[a]
                for z in range(2):
                    new_mass=[sum(mass[s]*t[s][j] for s in range(2))*o[j][z] for j in range(2)];prob=sum(new_mass)
                    outcomes.append({'observation':z,'state_mass':list(map(pack,new_mass)),
                                     'reach_probability':pack(prob),'conditional_probability':pack(prob/reach) if reach else None,
                                     'conditional_belief':list(map(pack,[x/prob for x in new_mass])) if prob else None})
                    following.append((history+[z],new_mass,node['children'][z]))
            records.append({'observation_history':history[:],'state_mass':list(map(pack,mass)),
                            'reach_probability':pack(reach),'conditional_belief':list(map(pack,[x/reach for x in mass])) if reach else None,
                            'action':a,'reward_mass':pack(reward),'conditional_reward':pack(reward/reach) if reach else None,
                            'discounted_reward_mass':pack(weighted),'outcomes':outcomes})
        total_mass=[sum(mass[s] for _,mass,_ in current) for s in range(2)]
        if sum(total_mass)!=1:raise ArithmeticError('history occupancy mass invariant failed')
        levels.append({'time':depth,'discount_weight':pack(discount**depth),'nodes':records,
                       'total_state_mass':list(map(pack,total_mass)),'total_reach_probability':pack(sum(total_mass)),
                       'discounted_reward_mass':pack(level_reward)})
        value+=level_reward;current=following
    alpha_value=sum(b[s]*alpha[s] for s in range(2))
    if value!=alpha_value:raise ArithmeticError('forward reward and backward policy alpha disagree')
    return {'format':'belieflab-policy-evaluation-v1','model':canonical,'belief':list(map(pack,b)),
            'horizon':horizon,'policy_tree':tree,'policy_alpha':list(map(pack,alpha)),'value':pack(value),
            'alpha_value':pack(alpha_value),'levels':levels,'decision_histories':2**horizon-1,
            'terminal_histories':2**horizon,'positive_decision_histories':positive_decisions,
            'scope':'fixed supplied observation-contingent tree; current-state reward then transition then new-state emission; unconditional state/history masses sum to one at every time including terminal; conditional beliefs and conditional rewards are null on unreachable histories, not fabricated; all leaves must have equal horizon0..4; at most15 decision histories and16 terminal histories, not time/RAM bounds; exact expectation in supplied model, not realized return, calibration or real-world safety'}


def _components(pieces):
    merged=[]
    for lo,hi in sorted(pieces):
        if merged and lo<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],hi)
        else:merged.append([lo,hi])
    return [[pack(x) for x in p] for p in merged]


def audit_policy(model,belief,policy_tree,*,candidate_limit=4096):
    """Certify supplied-tree regret against the optimized same-horizon value."""
    evaluation=evaluate_policy(model,belief,policy_tree)
    optimum=solve(model,belief,horizon=evaluation['horizon'],candidate_limit=candidate_limit)
    fixed=[F(*x) for x in evaluation['policy_alpha']];segments=[];zeros=[];knots={}
    for n in optimum['layers'][-1]['nodes']:
        lo,hi=map(lambda x:F(*x),n['belief0_interval']);delta=[F(*n['alpha'][s])-fixed[s] for s in range(2)]
        gap=[x*delta[0]+(1-x)*delta[1] for x in [lo,hi]]
        if min(gap)<0:raise ArithmeticError('policy regret dominance invariant failed')
        if gap==[0,0]:zeros.append((lo,hi))
        else:zeros.extend((x,x) for x,g in zip([lo,hi],gap) if g==0)
        knots[lo]=gap[0];knots[hi]=gap[1]
        segments.append({'belief0_interval':[pack(lo),pack(hi)],'optimal_node_id':n['id'],
                         'regret_alpha':list(map(pack,delta)),'regret_endpoint_values':list(map(pack,gap))})
    maximum=max(knots.values());maxima=[(x,x) for x,v in knots.items() if v==maximum]
    maxima.extend((F(*n['belief0_interval'][0]),F(*n['belief0_interval'][1])) for n in segments if list(map(lambda x:F(*x),n['regret_endpoint_values']))==[maximum,maximum])
    regret=F(*optimum['value'])-F(*evaluation['value'])
    return {'format':'belieflab-policy-audit-v1','evaluation':evaluation,'optimal_solution':optimum,
            'regret_at_input_belief':pack(regret),'optimal_at_input_belief':regret==0,
            'regret_segments':segments,'optimality_belief0_intervals':_components(zeros),
            'maximum_regret':pack(maximum),'maximum_regret_belief0_intervals':_components(maxima),
            'scope':'supplied tree compared with optimal observation-contingent value in SAME model, horizon, discount and rewards; exact piecewise-affine nonnegative regret on every belief; zero-regret intervals include isolated points; fixed policy alpha is linear so regret is convex and its maximum occurs at a belief-domain endpoint (possibly elsewhere in a flat maximum); not a learned-policy statistical certificate or real-world performance guarantee'}
