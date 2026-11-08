"""Policy mass/reward accounting and exact supplied-policy regret."""
from .demo import sensing_model
from .policy_audit import audit_policy


def policy_demo():
    model=sensing_model();prior=[[1,2],[1,2]]
    exploit0={'action':0,'children':[None,None]};exploit1={'action':1,'children':[None,None]}
    contingent={'action':2,'children':[exploit0,exploit1]}
    fixed={'action':0,'children':[exploit0,exploit0]}
    good=audit_policy(model,prior,contingent);bad=audit_policy(model,prior,fixed)
    e=good['evaluation']
    assert e['value']==[7,10] and e['policy_alpha']==[[7,10],[7,10]]
    assert [n['discounted_reward_mass'] for n in e['levels']]==[[-1,10],[4,5],[0,1]]
    assert [n['reach_probability'] for n in e['levels'][1]['nodes']]==[[1,2],[1,2]]
    assert [n['discounted_reward_mass'] for n in e['levels'][1]['nodes']]==[[2,5],[2,5]]
    assert good['optimality_belief0_intervals']==[[[13,40],[27,40]]]
    assert good['maximum_regret']==[13,10] and good['maximum_regret_belief0_intervals']==[[[0,1],[0,1]],[[1,1],[1,1]]]
    assert bad['evaluation']['value']==[0,1] and bad['regret_at_input_belief']==[7,10]
    assert bad['maximum_regret']==[4,1] and bad['optimality_belief0_intervals']==[[[27,40],[1,1]]]
    return {'contingent_policy':good,'fixed_exploit0_policy':bad,
            'selection':'same directly specified two-state sensing model; compare supplied probe-then-follow-indication tree and supplied exploit0-twice tree, both horizon2; fixed tree is an arbitrary policy witness, NOT the optimized open-loop envelope from feedback-demo',
            'scope':'forward exact probability/reward expectation, not sampled execution; optimal at one prior need not be optimal at every prior; posterior beliefs describe the supplied model, not empirical calibration'}
