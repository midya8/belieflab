"""Exact finite-horizon belief envelopes and observation-contingent branches."""
import argparse,html,json
from pathlib import Path
from belieflab.demo import demo


def render(d):
    if d!=demo():raise ValueError('requires complete unchanged demo')
    nodes=d['two_step']['layers'][-1]['nodes'];out=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="780" viewBox="0 0 1120 780">','<rect width="1120" height="780" fill="#101827"/>','<style>text{font-family:Arial,sans-serif;fill:#e7edf8}</style>','<metadata>'+html.escape(json.dumps(d,sort_keys=True))+'</metadata>']
    def text(x,y,s,size=16,color='#e7edf8'):out.append(f'<text x="{x}" y="{y}" font-size="{size}" style="fill:{color}">'+html.escape(s)+'</text>')
    text(30,43,'BeliefLab | paying for information can be the best first action',28)
    text(30,80,'Two hidden states. Exact rational policy trees and a finite-horizon envelope over every belief.',17)
    text(30,124,'Two-step optimal value',21);text(750,124,'At the uniform belief',21)
    for tick in [0,.5,1,1.5,2]:
        y=500-160*tick;out.append(f'<path d="M 90 {y:g} H 660" stroke="#33445e"/>');text(35,y+5,str(tick),15)
    for tick in [0,.25,.5,.75,1]:
        x=90+570*tick;text(x-12,530,str(tick),15)
    out.append('<path d="M 90 180 L 275.25 388 L 474.75 388 L 660 180" data-envelope="horizon2" fill="none" stroke="#63d8bc" stroke-width="5"/>')
    for x,v in [(275.25,'13/40'),(474.75,'27/40')]:out.append(f'<path d="M {x} 388 V 500" data-breakpoint="{v}" stroke="#e9bb79" stroke-dasharray="6 5"/>')
    text(295,368,'probe value = 7/10',19,'#63d8bc');text(220,560,'13/40',15,'#e9bb79');text(470,560,'27/40',15,'#e9bb79')
    text(95,590,'b = probability of hidden state 0',17)
    text(750,176,'Best two-step value by first action:',17)
    for y,label,value in [(216,'Exploit state 0','0'),(252,'Exploit state 1','0'),(288,'Probe','7/10')]:text(750,y,label,18);text(1010,y,value,20,'#63d8bc' if value=='7/10' else '#e7edf8')
    text(750,343,'Probe immediate reward: -1/10.',17)
    text(750,380,'Observation 0: probability 1/2',17)
    text(750,409,'posterior (9/10, 1/10), exploit 0.',16,'#b7c4d9')
    text(750,452,'Observation 1: probability 1/2',17)
    text(750,481,'posterior (1/10, 9/10), exploit 1.',16,'#b7c4d9')
    text(750,536,'Expected next reward = 4/5.',17)
    text(750,568,'Total: -1/10 + 4/5 = 7/10.',17,'#63d8bc')
    text(30,635,'Probe is optimal for b in [13/40, 27/40], with a boundary tie. Outside: exploit the more likely state.',17)
    text(30,673,'Same two-step horizon for every root-action comparison. The myopic one-step exploit reward is zero at b=1/2.',16)
    text(30,710,'Exact envelope, not a belief grid. Duplicate/redundant tied trees may be pruned; root-action values remain exact.',16)
    text(30,747,'Supplied two-state finite-horizon model only, not learned sensing, infinite-horizon convergence or real-world safety.',16,'#b7c4d9')
    text(30,775,'Full exact model, all layers, selected policy trees, root values and Bayesian branch certificates are in SVG metadata.',14,'#b7c4d9')
    return '\n'.join(out+['</svg>'])+'\n'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args();s=render(demo());path=Path(__file__).resolve().parents[1]/'assets'/'belief-audit.svg'
    if a.check:
        if path.read_text()!=s:raise SystemExit('SVG differs')
    else:path.parent.mkdir(exist_ok=True);path.write_text(s)
