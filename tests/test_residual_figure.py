import copy,json,unittest,xml.etree.ElementTree as ET
from fractions import Fraction as F
from pathlib import Path
from belieflab.residual_demo import residual_demo
from examples.render_residual import render
NS={'s':'http://www.w3.org/2000/svg'}

class ResidualFigureTests(unittest.TestCase):
    def test_all_curve_vertices_exact_breakpoints_and_shared_scale(self):
        d=residual_demo();r=d['perfect_probe'];root=ET.fromstring(render(d));curves={n.attrib['data-curve']:n for n in root.findall('s:polyline',NS)}
        points=[(F(*r['segments'][0]['belief0_interval'][0]),F(*r['segments'][0]['base_endpoint_values'][0]))]+[(F(*s['belief0_interval'][1]),F(*s['base_endpoint_values'][1])) for s in r['segments']]
        expected={'base':points,'lower':[(x,y+F(31,300)) for x,y in points],'upper':[(x,y+F(1,4)) for x,y in points],'exact':[(F(0),F(2)),(F(11,40),F(9,10)),(F(29,40),F(9,10)),(F(1),F(2))]}
        self.assertEqual(set(curves),set(expected))
        for key,vals in expected.items():
            actual=[tuple(map(float,q.split(','))) for q in curves[key].attrib['points'].split()];self.assertEqual(len(actual),len(vals))
            for (x,y),(b,v) in zip(actual,vals):self.assertAlmostEqual(x,float(90+600*b),places=8);self.assertAlmostEqual(y,float(635-150*v),places=8)
        self.assertEqual(curves['exact'].attrib['stroke-dasharray'],'8 5')
    def test_full_metadata_and_explicit_value_not_policy_scope(self):
        root=ET.fromstring(render(residual_demo()));self.assertEqual(json.loads(root.find('s:metadata',NS).text),residual_demo());text=' '.join(root.itertext());self.assertIn('not an infinite-horizon policy',text);self.assertIn('Discount 1 is refused',text);self.assertIn('[113/150, 9/10]',text)
    def test_regeneration(self):self.assertEqual((Path(__file__).resolve().parents[1]/'assets'/'discounted-certificate.svg').read_text(),render(residual_demo()))
    def test_reject_changed_undisplayed_negative_fixture_or_lookahead(self):
        d=copy.deepcopy(residual_demo());d['negative_constant']['minimum_residual'][0]+=1
        with self.assertRaises(ValueError):render(d)
        d=copy.deepcopy(residual_demo());d['perfect_probe']['lookahead_solution']['layers'][1]['nodes'][0]['alpha'][0][0]+=1
        with self.assertRaises(ValueError):render(d)
