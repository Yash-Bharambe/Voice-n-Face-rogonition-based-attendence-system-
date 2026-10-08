import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.pipelines import voice_challenge as vc


class VoiceChallengeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path_patch = patch.object(vc, 'STATE_PATH', Path(self.temp.name) / 'state.sqlite3')
        self.path_patch.start()
        self.code = 'apple river silver garden'
        self.students = {
            '23': {'student_id': 1, 'voice_embedding': [1., 0.]},
            '24': {'student_id': 2, 'voice_embedding': [0., 1.]},
        }

    def tearDown(self):
        self.path_patch.stop()
        self.temp.cleanup()

    def phrase(self, roll='two three', code=None):
        return f'Roll number {roll}, code {code or self.code}.'

    def test_matching_voice_and_code_required(self):
        self.assertEqual(vc.verify_claim(self.phrase(), self.code, [1., 0.], self.students), (1, 1., '23'))
        for phrase, embedding in [(self.phrase(code='apple river winter sunset'), [1., 0.]),
                                  (self.phrase(), [0., 1.]),
                                  (self.phrase('99'), [1., 0.]),
                                  ('I am present', [1., 0.])]:
            with self.assertRaises(ValueError):
                vc.verify_claim(phrase, self.code, embedding, self.students)

    def test_ambiguous_voice_is_rejected(self):
        self.students['24']['voice_embedding'] = [0.999, 0.001]
        with self.assertRaises(ValueError):
            vc.verify_claim(self.phrase(), self.code, [1., 0.], self.students)

    def test_multiple_claims_or_trailing_text_rejected(self):
        for text in [self.phrase() + ' ' + self.phrase('24'), self.phrase() + ' hello',
                     self.phrase(code='garden silver river apple')]:
            with self.assertRaises(ValueError):
                vc.parse_claim(text, self.code)

    def test_numeric_roll_formats(self):
        for value in ['23', '2 3', 'two three', 'twenty three', 'zero two three']:
            self.assertEqual(vc.parse_claim(self.phrase(value), self.code), '23')
        self.assertEqual(vc.parse_claim('Role number 2-3. Code Apple River Silver Garden.', self.code), '23')

    def test_screenshot_transcript_without_code_separator(self):
        code = 'river pencil button sunset'
        self.assertEqual(vc.parse_claim('Roll number 62, River Pencil Button Sunset', code), '62')
        self.assertEqual(vc.parse_claim('Roll number sixty two, River Pencil Button Sunset', code), '62')

    def test_code_misheard_as_cold(self):
        code = 'river pencil button sunset'
        transcript = 'Roll number 62 cold river pencil button sunset'
        students = {'62': {'student_id': 1, 'voice_embedding': [1., 0.]}}
        self.assertEqual(vc.verify_claim(transcript, code, [1., 0.], students), (1, 1., '62'))
        for bad in [transcript.replace('button sunset', 'apple garden'),
                    transcript + ' hello', transcript.replace('62', '63')]:
            with self.assertRaises(ValueError):
                vc.verify_claim(bad, code, [1., 0.], students)
        with self.assertRaises(ValueError):
            vc.verify_claim(transcript, code, [0., 1.], students)

    def test_omitted_separator_keeps_all_verification_gates(self):
        phrase = 'Roll number 23, Apple River Silver Garden'
        self.assertEqual(vc.verify_claim(phrase, self.code, [1., 0.], self.students), (1, 1., '23'))
        for text in [phrase.replace('Silver Garden', 'Winter Sunset'), phrase.replace('River Silver ', ''),
                     phrase + ' hello', phrase + ' ' + phrase, 'Roll number 23']:
            with self.assertRaises(ValueError):
                vc.parse_claim(text, self.code)
        with self.assertRaises(ValueError):
            vc.verify_claim(phrase, self.code, [0., 1.], self.students)

    def test_one_challenge_word_error_with_and_without_separator(self):
        words = self.code.split()
        for i in range(4):
            changed = words.copy()
            changed[i] = 'winter'
            omitted = words[:i] + words[i+1:]
            for spoken in (changed, omitted):
                for separator in ('code ', ''):
                    text = 'Roll number 23, ' + separator + ' '.join(spoken)
                    self.assertEqual(vc.verify_claim(text, self.code, [1., 0.], self.students), (1, 1., '23'))
                    with self.assertRaises(ValueError):
                        vc.verify_claim(text, self.code, [0., 1.], self.students)
                    with self.assertRaises(ValueError):
                        vc.verify_claim(text.replace('23', '99'), self.code, [1., 0.], self.students)

    def test_tolerance_rejects_two_errors_insertions_and_swaps(self):
        for code in ['winter river silver sunset', 'apple garden',
                     'apple river garden silver', 'apple river silver garden hello',
                     'apple winter river silver garden']:
            with self.assertRaises(ValueError):
                vc.parse_claim(self.phrase(code=code), self.code)

    def test_challenge_persists_and_rotation_invalidates_old_code(self):
        with patch.object(vc, 'today', return_value='2026-10-02'):
            first = vc.get_challenge(7)
            self.assertEqual(first, vc.get_challenge(7))
            second = vc.get_challenge(7, rotate=True)
            self.assertNotEqual(first[1], second[1])
            with self.assertRaises(ValueError):
                vc.parse_claim(self.phrase(code=first[1]), second[1])
        with patch.object(vc, 'today', return_value='2026-10-03'):
            self.assertEqual(vc.get_challenge(7)[0], '2026-10-03')

    def test_roll_conflicts_roll_back_and_swaps_work(self):
        vc.save_rolls({1: '023', 2: '24'})
        vc.save_rolls({1: '24', 2: '23'})
        self.assertEqual(vc.get_rolls(), {1: '24', 2: '23'})
        with self.assertRaises(ValueError):
            vc.save_rolls({3: '24'})
        self.assertEqual(vc.get_rolls(), {1: '24', 2: '23'})


if __name__ == '__main__':
    unittest.main()
