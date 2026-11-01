"""Public report hides validation; saved training evidence stays untouched."""
import json
import re
import unittest
from pathlib import Path
import make_report as report

ROOT = Path(__file__).parent
HIDDEN = re.compile(r'validation|cross[ -]validation|\bCV\b|fold|แถวตรวจ|ชุดตรวจสอบ|VAL_SIZE|N_FOLDS', re.I)


class ReportVisibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs = json.loads((ROOT / 'docs/results.json').read_text(encoding='utf-8'))['runs']

    def test_both_pages_hide_validation(self):
        for old in (False, True):
            with self.subTest(old=old):
                page, _ = report.render(self.runs, [], old=old)
                self.assertIsNone(HIDDEN.search(page))
                self.assertIn('Test', page)
                self.assertIn('ความสำคัญของตัวแปร', page)
                self.assertIn('เวลารัน', page)

    def test_main_page_preserves_test_scores(self):
        page, _ = report.render(self.runs, [])
        for value in ('51,271', '52,275', '56,628', '0.9458', '4,608', '36,859'):
            self.assertIn(value, page)
        self.assertNotIn('4,607', page)

    def test_public_export_hides_validation_without_mutating_input(self):
        before = json.dumps(self.runs, ensure_ascii=False)
        public = report.public_results(self.runs)
        self.assertEqual(before, json.dumps(self.runs, ensure_ascii=False))
        self.assertIsNone(HIDDEN.search(json.dumps(public, ensure_ascii=False)))
        for original, exported in zip(self.runs, public):
            self.assertEqual(original['result']['test'], exported['result']['test'])
            self.assertNotIn('validation', exported['result'])


if __name__ == '__main__':
    unittest.main()
