from .core import solve,belief_update


def sensing_model():
    identity=[[1,0],[0,1]];uninformative=[[[1,2],[1,2]],[[1,2],[1,2]]]
    return {'version':1,'discount':1,'actions':[
        {'reward':[1,-1],'transition':identity,'observation':uninformative},
        {'reward':[-1,1],'transition':identity,'observation':uninformative},
        {'reward':[[-1,10],[-1,10]],'transition':identity,'observation':[[[9,10],[1,10]],[[1,10],[9,10]]]}]}


def demo():
    model=sensing_model();myopic=solve(model,[[1,2],[1,2]],horizon=1);planned=solve(model,[[1,2],[1,2]],horizon=2)
    branches=[belief_update(model,[[1,2],[1,2]],2,z) for z in range(2)]
    assert myopic['value']==[0,1] and myopic['action'] in [0,1]
    assert planned['action']==2 and planned['value']==[7,10]
    assert planned['root_action_values']==[[0,1],[0,1],[7,10]] and planned['optimal_root_actions']==[2]
    assert branches[0]['posterior']==[[9,10],[1,10]] and branches[1]['posterior']==[[1,10],[9,10]]
    assert planned['policy_tree']=={'action':2,'children':[{'action':0,'children':[None,None]},{'action':1,'children':[None,None]}]}
    intervals=[n['belief0_interval'] for n in planned['layers'][-1]['nodes']]
    assert intervals==[[[0,1],[13,40]],[[13,40],[27,40]],[[27,40],[1,1]]]
    return {'model':model,'myopic':myopic,'two_step':planned,'probe_branches':branches,
            'expected_second_step_reward':[4,5],'probe_immediate_reward':[-1,10],
            'selection':'direct identity transitions; two exploit actions reward +/-1; probe costs1/10 and observes hidden state correctly9/10; uniform initial belief and undiscounted horizon2; no parameter search',
            'scope':'finite supplied two-state POMDP; information helps this prescribed horizon/reward model, not every sensor or environment; alpha envelope preserves values and exact root-action values at the input belief, not every tied policy tree'}
