import unittest
from unittest.mock import patch
import numpy as np

from src.pipelines.voice_overlap import verify_streams, separate_audio


class OverlapTests(unittest.TestCase):
    code = 'apple river silver garden'
    candidates = {
        '23': {'student_id': 1, 'voice_embedding': [1., 0., 0.]},
        '24': {'student_id': 2, 'voice_embedding': [0., 1., 0.]},
        '25': {'student_id': 3, 'voice_embedding': [0., 0., 1.]},
    }

    def claim(self, roll, voice, code=None):
        return f'Roll number {roll}, code {code or self.code}', voice

    def test_three_distinct_verified_students(self):
        claims = [self.claim(r, s['voice_embedding']) for r, s in self.candidates.items()]
        with patch('src.pipelines.voice_overlap.transcribe_waveform', side_effect=claims):
            results = verify_streams([np.ones(10)] * 3, self.code, self.candidates)
        self.assertEqual([r['student_id'] for r in results], [1, 2, 3])
        self.assertTrue(all(r['Status'] == 'Verified' for r in results))

    def test_wrong_code_and_impostor_stay_pending(self):
        claims = [self.claim('23', [1., 0., 0.]), self.claim('24', [0., 1., 0.], 'old wrong daily code'),
                  self.claim('25', [1., 0., 0.])]
        with patch('src.pipelines.voice_overlap.transcribe_waveform', side_effect=claims):
            results = verify_streams([np.ones(10)] * 3, self.code, self.candidates)
        self.assertEqual([r['Status'] for r in results], ['Verified', 'Retry', 'Retry'])
        self.assertIsNone(results[1]['student_id'])
        self.assertIsNone(results[2]['student_id'])

    def test_duplicate_student_leakage_rejects_both_streams(self):
        with patch('src.pipelines.voice_overlap.transcribe_waveform', return_value=self.claim('23', [1., 0., 0.])):
            results = verify_streams([np.ones(10)] * 2, self.code, self.candidates)
        self.assertTrue(all(r['Status'] == 'Retry' and r['student_id'] is None for r in results))

    def test_missing_source_and_unclear_source_rejected(self):
        with patch('src.pipelines.voice_overlap.transcribe_waveform', side_effect=ValueError('Unclear audio')):
            results = verify_streams([None, np.ones(10)], self.code, self.candidates)
        self.assertTrue(all(r['Status'] == 'Retry' for r in results))

    def test_unsupported_speaker_counts_rejected_before_loading(self):
        for count in (0, 1, 4, 5):
            with self.assertRaises(ValueError):
                separate_audio(b'', count)


if __name__ == '__main__':
    unittest.main()
