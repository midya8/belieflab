"""Reproducible full-belief discounted-value bracket from exact demo data."""
import argparse,html,json
from fractions import Fraction as F
from pathlib import Path
from belieflab.residual_demo import residual_demo


def render(d):
    if d!=residual_demo():raise ValueError('requires complete unchanged residual demo')
    out=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="780" viewBox="0 0 1120 780">','<rect width="1120" height="780" fill="#101827"/>','<style>text{font-family:Arial,sans-serif;fill:#e7edf8}</style>','<metadata>'+html.escape(json.dumps(d,sort_keys=True))+'</metadata>']
    def text(x,y,s,size=16,color='#e7edf8'):
        out.append(f'<text x="{x}" y="{y}" font-size="{size}" style="fill:{color}">'+html.escape(s)+'</text>')
    def coord(x):return format(float(x),'.9f').rstrip('0').rstrip('.')
    def line(name,points,color,dashed=False):
        coords=' '.join(coord(F(90)+600*x)+','+coord(F(635)-150*y) for x,y in points)
        out.append(f'<polyline data-curve="{name}" points="{coords}" fill="none" stroke="{color}" stroke-width="3"'+(' stroke-dasharray="8 5"' if dashed else '')+'/>')
    text(30,43,'BeliefLab | a small Bellman residual can hide a large value error',26)
    text(30,82,'Constant reward 1/100, discount 99/100, zero-decision base value 0. Exact, prescribed fixture.',17)
    for x,label,value in [(40,'Raw residual','1/100'),(400,'Divide by 1 - discount','1'),(800,'Exact infinite value','1')]:
        text(x,133,label,18,'#b7c4d9');text(x,179,value,32,'#63d8bc')
    text(30,220,'Signed bounds retain direction: reward -1/100 gives residual -1/100 and exact value -1.',17)
    out.append('<path d="M 30 245 H 1080" stroke="#33445e"/>')
    text(30,282,'A hidden-state example: bracket V* over every initial belief',23)
    text(30,312,'Static state. Perfect probe costs 1/10. Discount 1/2. Base horizon 3; one extra backup.',17)
    for y,label in [(F(0),'0'),(F(1),'1'),(F(2),'2')]:
        yy=635-150*y;out.append(f'<path d="M 90 {yy} H 690" stroke="#33445e"/>');text(58,yy+5,label,16,'#b7c4d9')
    for x,label in [(F(0),'0'),(F(1,2),'1/2'),(F(1),'1')]:text(float(90+600*x)-8,663,label,16,'#b7c4d9')
    r=d['perfect_probe'];segs=r['segments'];points=[]
    for i,s in enumerate(segs):
        if i==0:points.append((F(*s['belief0_interval'][0]),F(*s['base_endpoint_values'][0])))
        points.append((F(*s['belief0_interval'][1]),F(*s['base_endpoint_values'][1])))
    lower,upper=map(lambda v:F(*v),r['signed_value_error_interval'])
    line('base',points,'#98a8c0');line('lower',[(x,y+lower) for x,y in points],'#e9bb79');line('upper',[(x,y+upper) for x,y in points],'#63d8bc')
    line('exact',[(F(0),F(2)),(F(11,40),F(9,10)),(F(29,40),F(9,10)),(F(1),F(2))],'#c4a2ff',True)
    for y,label,color in [(350,'Gray: finite value V3','#98a8c0'),(381,'Amber: certified lower bound','#e9bb79'),(412,'Green: certified upper bound','#63d8bc'),(443,'Dashed: analytic exact V*','#c4a2ff')]:text(735,y,label,16,color)
    text(735,490,'Global residual: [31/600, 1/8]',17)
    text(735,522,'Signed shifts: [31/300, 1/4]',17)
    text(735,563,'At b = 1/2: V3 = 13/20.',18)
    text(735,595,'Certificate: [113/150, 9/10].',18,'#63d8bc')
    text(735,627,'Exact infinite value: 9/10.',18,'#c4a2ff')
    text(90,701,'b = probability of hidden state 0. Piecewise lines use exact rational breakpoints, not a belief grid.',16)
    text(30,741,'Value bracket only: the finite tree is not an infinite-horizon policy. Discount 1 is refused.',17)
    text(30,769,'Supplied-model guarantee, not learned-model validity, action optimality, policy-loss or real-world safety.',16,'#b7c4d9')
    return '\n'.join(out+['</svg>'])+'\n'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args();s=render(residual_demo());path=Path(__file__).resolve().parents[1]/'assets'/'discounted-certificate.svg'
    if a.check:
        if path.read_text()!=s:raise SystemExit('SVG differs')
    else:path.parent.mkdir(exist_ok=True);path.write_text(s)
