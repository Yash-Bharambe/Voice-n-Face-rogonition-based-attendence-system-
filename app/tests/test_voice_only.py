import unittest
from unittest.mock import patch
import numpy as np
from src.pipelines.voice_challenge import verify_voice
from src.pipelines.voice_overlap import verify_streams


class VoiceOnlyTests(unittest.TestCase):
    profiles = {'1': {'student_id': 1, 'voice_embedding': [1., 0.]},
                '2': {'student_id': 2, 'voice_embedding': [0., 1.]}}

    def test_identification_and_unknown_voice(self):
        self.assertEqual(verify_voice([1., 0.], self.profiles), (1, 1., '1'))
        for voice in ([0., 0.], [-1., -1.], [1., 1.]):
            with self.assertRaises(ValueError):
                verify_voice(voice, self.profiles)

    def test_ambiguous_voice(self):
        profiles = {**self.profiles, '3': {'student_id': 3, 'voice_embedding': [1., .01]}}
        with self.assertRaises(ValueError):
            verify_voice([1., 0.], profiles)

    def test_overlap_does_not_transcribe_and_rejects_duplicates(self):
        with patch('src.pipelines.voice_overlap.transcribe_waveform') as transcribe, \
             patch('src.pipelines.voice_overlap.voice_embedding_waveform', side_effect=[[1., 0.], [0., 1.]]):
            results = verify_streams([np.ones(10)] * 2, None, self.profiles, voice_only=True)
        transcribe.assert_not_called()
        self.assertEqual([r['student_id'] for r in results], [1, 2])
        with patch('src.pipelines.voice_overlap.voice_embedding_waveform', return_value=[1., 0.]):
            results = verify_streams([np.ones(10)] * 2, None, self.profiles, voice_only=True)
        self.assertTrue(all(r['Status'] == 'Retry' for r in results))
