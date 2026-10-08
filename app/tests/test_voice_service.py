import json
import subprocess
import unittest
from unittest.mock import Mock, patch

from src.pipelines import voice_service as service


class VoiceServiceTests(unittest.TestCase):
    def worker(self, code=0, result=None):
        process = Mock(returncode=code)
        process.communicate.return_value = (json.dumps(result or {'result': ['heard', [1.0]]}).encode(), b'')
        return process

    @patch.object(service.subprocess, 'Popen')
    def test_success_and_audio_stays_in_pipe(self, launch):
        launch.return_value = self.worker()
        self.assertEqual(service.transcribe_claim(b'voice'), ('heard', [1.0]))
        payload = json.loads(launch.return_value.communicate.call_args.args[0])
        self.assertEqual(payload['audio'], 'dm9pY2U=')
        self.assertNotIn('voice', launch.call_args.args[0])

    @patch.object(service.subprocess, 'Popen')
    def test_native_crash_becomes_recoverable_error(self, launch):
        launch.return_value = self.worker(code=3221225477)
        with self.assertRaisesRegex(ValueError, 'app is still running'):
            service.transcribe_claim(b'voice')

    @patch.object(service.subprocess, 'Popen')
    def test_timeout_kills_and_reaps_worker(self, launch):
        process = self.worker()
        process.communicate.side_effect = [subprocess.TimeoutExpired('worker', 600), (b'', b'')]
        launch.return_value = process
        with self.assertRaisesRegex(ValueError, 'too long'):
            service.transcribe_claim(b'voice')
        process.kill.assert_called_once()
        self.assertEqual(process.communicate.call_count, 2)

    @patch.object(service.subprocess, 'Popen')
    def test_validation_error_is_preserved(self, launch):
        launch.return_value = self.worker(result={'error': 'Speech was unclear.'})
        with self.assertRaisesRegex(ValueError, 'Speech was unclear'):
            service.transcribe_claim(b'voice')

    @patch.object(service.subprocess, 'Popen')
    def test_overlap_sends_only_needed_profile_fields(self, launch):
        launch.return_value = self.worker(result={'result': []})
        service.verify_overlapping_audio(b'voice', 2, 'apple river',
            {'23': {'student_id': 'student', 'voice_embedding': [1.0], 'password': 'private'}})
        payload = json.loads(launch.return_value.communicate.call_args.args[0])
        self.assertEqual(set(payload['candidates']['23']), {'student_id', 'voice_embedding'})


if __name__ == '__main__':
    unittest.main()
