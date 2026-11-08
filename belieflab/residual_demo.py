"""Prescribed discount-amplification and hidden-state bracket fixtures."""
from .demo import sensing_model
from .residual import certify_discounted


def residual_demo():
    constant={'version':1,'discount':[99,100],'actions':[{'reward':[[1,100],[1,100]],'transition':[[1,0],[0,1]],'observation':[[1,0],[1,0]]}]}
    positive=certify_discounted(constant,[[1,2],[1,2]],horizon=0)
    assert positive['residual_sup_norm']==[1,100] and positive['infinite_value_interval_at_input_belief']==[[1,1],[1,1]]
    negative={'version':1,'discount':[99,100],'actions':[dict(constant['actions'][0],reward=[[-1,100],[-1,100]])]}
    negative=certify_discounted(negative,[[1,2],[1,2]],horizon=0)
    assert negative['infinite_value_interval_at_input_belief']==[[-1,1],[-1,1]]
    m=sensing_model();m['discount']=[1,2];m['actions'][2]['observation']=[[1,0],[0,1]]
    sensing=certify_discounted(m,[[1,2],[1,2]],horizon=3)
    assert sensing['value_at_input_belief']==[13,20]
    assert sensing['infinite_value_interval_at_input_belief'][1]==[9,10]
    return {'format':'belieflab-residual-demo-v1','positive_constant':positive,'negative_constant':negative,'perfect_probe':sensing,
            'perfect_probe_exact_infinite_envelope':[{'alpha':[[2,1],[-2,1]]},{'alpha':[[-2,1],[2,1]]},{'alpha':[[9,10],[9,10]]}],
            'selection':'Prescribed fixtures, not searched performance benchmarks. Constant +/-1/100 reward with discount99/100 has exact infinite value +/-1. Static hidden state, uninformative exploit observations, perfect probe, discount1/2 and probe cost1/10 gives exact V*=max(2*abs(2*b-1),9/10). Exploit forever or probe once then exploit a revealed state attains each term; the displayed maximum satisfies the belief Bellman fixed-point equation.'}
