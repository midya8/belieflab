import copy,json,unittest,xml.etree.ElementTree as ET
from pathlib import Path
from examples.render_audit import render
from belieflab.demo import demo
NS={'s':'http://www.w3.org/2000/svg'}


class FigureTests(unittest.TestCase):
    def test_exact_envelope_vertices_and_breakpoints(self):
        root=ET.fromstring(render(demo()));paths=root.findall('s:path',NS);curve=next(p for p in paths if p.attrib.get('data-envelope')=='horizon2');self.assertEqual(curve.attrib['d'],'M 90 180 L 275.25 388 L 474.75 388 L 660 180')
        # x=90+570*b, y=500-160*value. No fitted or sampled plot.
        self.assertEqual([(p.attrib['data-breakpoint'],p.attrib['d']) for p in paths if 'data-breakpoint' in p.attrib],[('13/40','M 275.25 388 V 500'),('27/40','M 474.75 388 V 500')])
    def test_complete_exact_metadata(self):self.assertEqual(json.loads(ET.fromstring(render(demo())).find('s:metadata',NS).text),demo())
    def test_regeneration(self):self.assertEqual((Path(__file__).resolve().parents[1]/'assets'/'belief-audit.svg').read_text(),render(demo()))
    def test_reject_changed_undisplayed_node(self):
        d=copy.deepcopy(demo());d['two_step']['layers'][0]['nodes'][0]['alpha'][0][0]+=1
        with self.assertRaises(ValueError):render(d)
