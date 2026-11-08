"""Prescribed feedback/no-feedback comparison, with a zero-gap control."""
from .demo import sensing_model
from .open_loop import compare_feedback


def feedback_demo():
    model=sensing_model();prior=[[1,2],[1,2]]
    sensing=compare_feedback(model,prior,horizon=2)
    control=sensing_model()
    for action in control['actions']:
        action['observation']=[[[1,2],[1,2]],[[1,2],[1,2]]]
    no_information=compare_feedback(control,prior,horizon=2)
    assert sensing['contingent_solution']['value']==[7,10]
    assert sensing['open_loop_solution']['value']==[0,1]
    assert sensing['open_loop_solution']['selected_sequence']==[0,0]
    assert len(sensing['open_loop_solution']['sequences'])==9
    assert sensing['maximum_advantage']==[7,10]
    assert sensing['maximum_advantage_belief0_intervals']==[[[1,2],[1,2]]]
    assert sensing['uniform_belief_integral']==[49,400]
    assert no_information['maximum_advantage']==[0,1]
    assert no_information['maximum_advantage_belief0_intervals']==[[[0,1],[1,1]]]
    assert no_information['uniform_belief_integral']==[0,1]
    return {'sensing':sensing,'uninformative_control':no_information,
            'selection':'same identity-transition two-step +/-1 exploit model and probe cost1/10 as original demo; compare EVERY one of nine fixed action sequences against contingent optimum; control replaces every observation row by (1/2,1/2) and changes nothing else; chosen directly, no parameter search',
            'scope':'exact within each supplied model; control is a declared uninformative-emission check, not a real sensor experiment; uniform-belief integral is geometric area, not a measured distribution of deployed beliefs'}
