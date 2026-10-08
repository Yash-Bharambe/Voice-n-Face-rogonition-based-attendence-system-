"""Private subprocess entry point; stdout is exclusively the JSON response."""
import base64
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    request = json.load(sys.stdin)
    audio = base64.b64decode(request['audio'], validate=True)
    try:
        # Some model libraries print startup messages. Keep the protocol intact.
        with redirect_stdout(sys.stderr):
            if request['operation'] == 'identify':
                from src.pipelines.voice_pipeline import voice_embedding_bytes
                from src.pipelines.voice_challenge import verify_voice
                sid, score, key = verify_voice(voice_embedding_bytes(audio), request['candidates'])
                result = {'student_id': sid, 'score': score, 'roll': key}
            elif request['operation'] == 'identify_overlap':
                from src.pipelines.voice_overlap import separate_audio, verify_streams
                result = verify_streams(separate_audio(audio, request['speaker_count']), None, request['candidates'], voice_only=True)
            elif request['operation'] == 'overlap':
                from src.pipelines.voice_overlap import verify_overlapping_audio
                result = verify_overlapping_audio(audio, request['speaker_count'], request['code'], request['candidates'])
            elif request['operation'] == 'transcribe':
                from src.pipelines.voice_pipeline import transcribe_claim
                transcript, embedding = transcribe_claim(audio)
                result = [transcript, embedding.tolist()]
            elif request['operation'] == 'enroll':
                from src.pipelines.voice_pipeline import enroll_voice_sample
                result = enroll_voice_sample(audio)
            else:
                raise ValueError('Unsupported voice operation.')
        response = {'result': result}
    except ValueError as exc:
        response = {'error': str(exc)}
    json.dump(response, sys.stdout)


if __name__ == '__main__':
    main()
