"""Forward branch accounting and exact optimality ranges of two supplied trees."""
import argparse,html,json
from pathlib import Path
from belieflab.policy_demo import policy_demo


def render(d):
    if d!=policy_demo():raise ValueError('requires complete unchanged policy demo')
    e=d['contingent_policy']['evaluation'];rows=[e['levels'][0]['nodes'][0]]+e['levels'][1]['nodes']
    out=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="800" viewBox="0 0 1120 800">','<rect width="1120" height="800" fill="#101827"/>','<style>text{font-family:Arial,sans-serif;fill:#e7edf8}</style>','<metadata>'+html.escape(json.dumps(d,sort_keys=True))+'</metadata>']
    def text(x,y,s,size=16,color='#e7edf8',cell=None):
        attr=f' data-cell="{cell}"' if cell else ''
        out.append(f'<text x="{x}" y="{y}" font-size="{size}" style="fill:{color}"{attr}>'+html.escape(s)+'</text>')
    def rational(x):return str(x[0]) if x[1]==1 else str(x[0])+'/'+str(x[1])
    text(30,43,'BeliefLab | follow the probability mass, then add the rewards',27)
    text(30,81,'Supplied probe-then-exploit policy. Uniform prior. Two decisions, discount 1, identity transitions.',17)
    for x,label in [(30,'Observation history'),(250,'Reach probability'),(430,'Conditional belief'),(675,'Action'),(870,'Discounted reward mass')]:text(x,137,label,16,'#b7c4d9')
    for i,n in enumerate(rows):
        y=200+56*i;out.append(f'<path d="M 30 {y+20} H 1080" stroke="#33445e"/>')
        values=[str(n['observation_history']),rational(n['reach_probability']),'('+', '.join(map(rational,n['conditional_belief']))+')',{0:'exploit 0',1:'exploit 1',2:'probe'}[n['action']],rational(n['discounted_reward_mass'])]
        for j,(x,value) in enumerate(zip([30,250,430,675,955],values)):text(x,y,value,20,'#e9bb79' if i==0 and j==4 else ('#63d8bc' if j==4 else '#e7edf8'),f'r{i}c{j}')
    text(30,370,'Each observation branch: reach 1/2 * conditional reward 4/5 = reward mass 2/5.',18)
    text(30,407,'Forward value: -1/10 + 2/5 + 2/5 = 7/10 = belief dot policy alpha (7/10, 7/10).',18,'#63d8bc')
    text(30,462,'Where is this fixed probe tree optimal?',20)
    out.append('<rect x="90" y="490" width="600" height="22" fill="#33445e"/>')
    out.append('<rect data-interval="probe" x="285" y="490" width="210" height="22" fill="#63d8bc"/>')
    text(285,538,'13/40',16);text(467,538,'27/40',16)
    text(740,473,'Maximum regret: 13/10.',18)
    text(740,505,'Attained at b = 0 and b = 1.',17,'#b7c4d9')
    text(30,580,'Where is the fixed exploit-0-twice tree optimal?',20)
    out.append('<rect x="90" y="603" width="600" height="22" fill="#33445e"/>')
    out.append('<rect data-interval="exploit0" x="495" y="603" width="195" height="22" fill="#e9bb79"/>')
    for x,label in [(90,'0'),(285,'13/40'),(390,'1/2'),(495,'27/40'),(690,'1')]:text(x-12,654,label,15)
    text(740,565,'Value at uniform belief: 0; regret 7/10.',17)
    text(740,597,'Maximum regret: 4 at b = 0.',18,'#e9bb79')
    text(740,632,'This tree is NOT the optimized',16,'#b7c4d9');text(740,659,'open-loop envelope from feedback-demo.',16,'#b7c4d9')
    text(90,698,'b = probability of hidden state 0. Colored bands show every zero-regret belief, including boundaries.',16)
    text(30,740,'Terminal history mass sums to 1. Impossible branches have zero mass and null conditional beliefs/rewards.',16)
    text(30,780,'Exact expectation in the supplied model, not sampled execution, learned calibration or real-world safety.',16,'#b7c4d9')
    return '\n'.join(out+['</svg>'])+'\n'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args();s=render(policy_demo());path=Path(__file__).resolve().parents[1]/'assets'/'policy-audit.svg'
    if a.check:
        if path.read_text()!=s:raise SystemExit('SVG differs')
    else:path.parent.mkdir(exist_ok=True);path.write_text(s)
