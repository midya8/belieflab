import copy,json,unittest,xml.etree.ElementTree as ET
from fractions import Fraction as F
from pathlib import Path
from belieflab.feedback_demo import feedback_demo
from examples.render_feedback import render
NS={'s':'http://www.w3.org/2000/svg'}


class FeedbackFigureTests(unittest.TestCase):
    def test_all_exact_curve_vertices_and_shared_scales(self):
        root=ET.fromstring(render(feedback_demo()));paths={p.attrib['data-series']:p for p in root.findall('s:path',NS) if 'data-series' in p.attrib}
        expected={'open-loop':[(0,2),(F(13,40),F(7,10)),(F(1,2),0),(F(27,40),F(7,10)),(1,2)],'contingent':[(0,2),(F(13,40),F(7,10)),(F(1,2),F(7,10)),(F(27,40),F(7,10)),(1,2)],'advantage':[(0,0),(F(13,40),0),(F(1,2),F(7,10)),(F(27,40),0),(1,0)]}
        for name,vertices in expected.items():
            words=paths[name].attrib['d'].split();actual=[(F(words[i+1]),F(words[i+2])) for i in range(0,len(words),3)];zero,scale=(650,140) if name=='advantage' else (440,120)
            self.assertEqual(actual,[(90+540*x,zero-scale*y) for x,y in vertices])
        self.assertEqual(paths['uninformative-gap'].attrib['d'],'M 90 650 H 630')
    def test_complete_metadata_and_caveats(self):
        root=ET.fromstring(render(feedback_demo()));self.assertEqual(json.loads(root.find('s:metadata',NS).text),feedback_demo());text=' '.join(root.itertext());self.assertIn('not a measured belief average',text);self.assertIn('9 complete fixed action sequences',text)
    def test_regeneration(self):self.assertEqual((Path(__file__).resolve().parents[1]/'assets'/'feedback-audit.svg').read_text(),render(feedback_demo()))
    def test_reject_undisplayed_sequence_or_control_change(self):
        for key in ['sensing','uninformative_control']:
            d=copy.deepcopy(feedback_demo());d[key]['open_loop_solution']['sequences'][-1]['alpha'][0][0]+=1
            with self.assertRaises(ValueError):render(d)
