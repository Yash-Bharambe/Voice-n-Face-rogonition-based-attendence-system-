import unittest
import numpy as np
from src.pipelines.voice_audio import prepare_audio,audio_quality

class VoiceAudioTests(unittest.TestCase):
    def test_quiet_signal_boosted_without_changing_shape(self):
        audio=(0.001*np.sin(np.arange(16000)*0.1)).astype(np.float32)
        normalized,quality=prepare_audio(audio)
        self.assertGreater(audio_quality(normalized)['rms'],quality['rms'])
        self.assertEqual(normalized.shape,audio.shape)
        self.assertLessEqual(np.max(np.abs(normalized)),0.98)
        self.assertAlmostEqual(float(np.mean(normalized)),0.,places=6)

    def test_silence_dc_and_invalid_signal_rejected(self):
        for audio in [np.zeros(16000),np.ones(16000)*0.1,np.array([np.nan]),np.array([])]:
            with self.assertRaises(ValueError):prepare_audio(audio)

    def test_clipped_recording_rejected(self):
        with self.assertRaises(ValueError):prepare_audio(np.tile([-1.,1.],8000))

    def test_gain_is_bounded(self):
        audio=(0.01*np.sin(np.arange(16000)*0.1)).astype(np.float32)
        normalized,_=prepare_audio(audio)
        self.assertLessEqual(audio_quality(normalized)['rms']/audio_quality(audio)['rms'],10.0001)

if __name__=='__main__':unittest.main()
