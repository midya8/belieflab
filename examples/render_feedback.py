"""Render exact feedback/open-loop envelopes and their difference, not a grid fit."""
import argparse,html,json
from fractions import Fraction as F
from pathlib import Path
from belieflab.feedback_demo import feedback_demo


def render(d):
    if d!=feedback_demo():raise ValueError('requires complete unchanged feedback demo')
    r=d['sensing'];out=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="800" viewBox="0 0 1120 800">','<rect width="1120" height="800" fill="#101827"/>','<style>text{font-family:Arial,sans-serif;fill:#e7edf8}</style>','<metadata>'+html.escape(json.dumps(d,sort_keys=True))+'</metadata>']
    def text(x,y,s,size=16,color='#e7edf8'):out.append(f'<text x="{x}" y="{y}" font-size="{size}" style="fill:{color}">'+html.escape(s)+'</text>')
    def coords(kind,zero,scale):
        vertices=[]
        for n in r['segments']:
            vertices.extend(zip(n['belief0_interval'],n[kind+'_endpoint_values']))
        vertices=list(dict.fromkeys((F(*b),F(*v)) for b,v in vertices))
        return ' '.join(('M' if i==0 else 'L')+f' {float(90+540*b):g} {float(zero-scale*v):g}' for i,(b,v) in enumerate(vertices))
    text(30,43,'BeliefLab | how much does feedback change the decision?',28)
    text(30,81,'Every fixed action sequence is optimized. Same two-step model, rewards and discount.',17)
    text(30,126,'Optimal total reward',21);text(740,126,'At b = 1/2',21)
    for v in [0,F(1,2),1,F(3,2),2]:
        y=float(440-120*v);out.append(f'<path d="M 90 {y:g} H 630" stroke="#33445e"/>');text(40,y+5,str(v),15)
    out.append('<path data-series="open-loop" d="'+coords('open_loop',440,120)+'" fill="none" stroke="#e9bb79" stroke-width="5"/>')
    out.append('<path data-series="contingent" d="'+coords('contingent',440,120)+'" fill="none" stroke="#63d8bc" stroke-width="4"/>')
    text(250,337,'contingent: 7/10',17,'#63d8bc');text(277,428,'open-loop: 0',16,'#e9bb79')
    for b,label in [(0,'0'),(.325,'13/40'),(.5,'1/2'),(.675,'27/40'),(1,'1')]:text(90+540*b-15,470,label,15)
    text(740,178,'Contingent optimum',18,'#63d8bc');text(1030,178,'7/10',20,'#63d8bc')
    text(740,217,'Open-loop optimum',18,'#e9bb79');text(1030,217,'0',20,'#e9bb79')
    text(740,264,'9 complete fixed action sequences.',17)
    text(740,297,'4 sequences tie at this prior.',17)
    text(740,330,'Selected fixed sequence: exploit 0 twice.',16,'#b7c4d9')
    text(740,374,'Contingent policy: probe, then exploit',17)
    text(740,404,'the state indicated by its observation.',17)
    text(30,511,'Feedback advantage = contingent minus open-loop',20)
    out.append('<path d="M 90 650 H 630" stroke="#e9bb79" stroke-dasharray="7 5" stroke-width="3" data-series="uninformative-gap"/>')
    out.append('<path data-series="advantage" d="'+coords('advantage',650,140)+'" fill="none" stroke="#63d8bc" stroke-width="4"/>')
    text(37,654,'0',15);text(295,543,'peak 7/10',17,'#63d8bc')
    for b,label in [(0,'0'),(.325,'13/40'),(.5,'1/2'),(.675,'27/40'),(1,'1')]:text(90+540*b-15,678,label,15)
    text(740,515,'Maximum gap: 7/10 at b = 1/2.',18,'#63d8bc')
    text(740,554,'Exact uniform-belief integral: 49/400.',17)
    text(740,586,'This is area, not a measured belief average.',16,'#b7c4d9')
    text(740,628,'Uninformative-emission control:',17,'#e9bb79')
    text(740,659,'gap = 0 at every belief (dashed).',17,'#e9bb79')
    text(90,710,'b = probability of hidden state 0. Lines connect exact envelope breakpoints, not sampled beliefs.',16)
    text(30,751,'Fixed sequences cannot condition later actions on observations. Both optima include all action costs.',16)
    text(30,783,'Supplied finite model only. Not learned information value, a causal sensor experiment or real-world safety.',16,'#b7c4d9')
    return '\n'.join(out+['</svg>'])+'\n'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args();s=render(feedback_demo());path=Path(__file__).resolve().parents[1]/'assets'/'feedback-audit.svg'
    if a.check:
        if path.read_text()!=s:raise SystemExit('SVG differs')
    else:path.parent.mkdir(exist_ok=True);path.write_text(s)
