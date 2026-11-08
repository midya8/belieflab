import copy,json,unittest,xml.etree.ElementTree as ET
from pathlib import Path
from belieflab.policy_demo import policy_demo
from examples.render_policy import render
NS={'s':'http://www.w3.org/2000/svg'}


class PolicyFigureTests(unittest.TestCase):
    def test_all15_table_cells_and_exact_shared_scale_intervals(self):
        root=ET.fromstring(render(policy_demo()));cells={n.attrib['data-cell']:n.text for n in root.findall('s:text',NS) if 'data-cell' in n.attrib}
        expected=[['[]','1','(1/2, 1/2)','probe','-1/10'],['[0]','1/2','(9/10, 1/10)','exploit 0','2/5'],['[1]','1/2','(1/10, 9/10)','exploit 1','2/5']]
        self.assertEqual(cells,{f'r{i}c{j}':v for i,row in enumerate(expected) for j,v in enumerate(row)})
        rects={n.attrib['data-interval']:n.attrib for n in root.findall('s:rect',NS) if 'data-interval' in n.attrib};self.assertEqual([(rects[k]['x'],rects[k]['width']) for k in ['probe','exploit0']],[('285','210'),('495','195')])
        # Shared x=90+600*b: [13/40,27/40] and [27/40,1], no sampled boundaries.
    def test_full_metadata_and_fixed_tree_scope(self):
        root=ET.fromstring(render(policy_demo()));self.assertEqual(json.loads(root.find('s:metadata',NS).text),policy_demo());text=' '.join(root.itertext());self.assertIn('NOT the optimized',text);self.assertIn('null conditional beliefs/rewards',text)
    def test_regeneration(self):self.assertEqual((Path(__file__).resolve().parents[1]/'assets'/'policy-audit.svg').read_text(),render(policy_demo()))
    def test_reject_changed_undisplayed_terminal_or_other_policy(self):
        for key in ['contingent_policy','fixed_exploit0_policy']:
            d=copy.deepcopy(policy_demo());d[key]['evaluation']['levels'][-1]['nodes'][-1]['state_mass'][0][0]+=1
            with self.assertRaises(ValueError):render(d)
