"""Experimental two/three-speaker separation with the same claim verification gates."""
import io
from collections import Counter
from threading import Lock

import librosa
import numpy as np
import streamlit as st

from src.pipelines.voice_pipeline import MODEL_DIRECTORY, transcribe_waveform, voice_embedding_waveform
from src.pipelines.voice_challenge import verify_claim, verify_voice

SEPARATION_LOCK = Lock()


@st.cache_resource
def load_separator(speaker_count):
    if speaker_count not in (2, 3):
        raise ValueError('Overlap mode supports exactly two or three speakers.')
    import torch
    from speechbrain.inference.separation import SepformerSeparation
    from speechbrain.utils.fetching import LocalStrategy
    torch.set_num_threads(4)
    directory = MODEL_DIRECTORY / f'sepformer-{speaker_count}'
    # COPY avoids Windows symlink permissions. Models stay in the ignored local directory.
    return SepformerSeparation.from_hparams(
        source=f'speechbrain/sepformer-wsj0{speaker_count}mix',
        savedir=str(directory), run_opts={'device': 'cpu'},
        local_strategy=LocalStrategy.COPY,
        huggingface_cache_dir=str(MODEL_DIRECTORY / 'speechbrain-cache'))


def separate_audio(audio_bytes, speaker_count):
    if speaker_count not in (2, 3):
        raise ValueError('Use groups of two or three students; four or five are not supported.')
    audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=8000, mono=True)
    if not 3 <= len(audio) / 8000 <= 12:
        raise ValueError('Record a group for 3–12 seconds.')
    if not np.isfinite(audio).all() or np.max(np.abs(audio)) < 0.005:
        raise ValueError('No usable speech was recorded.')
    import torch
    with SEPARATION_LOCK, torch.inference_mode():
        sources = load_separator(speaker_count).separate_batch(torch.from_numpy(audio).unsqueeze(0))
        sources = sources.squeeze(0).cpu().numpy()
    if sources.ndim != 2 or sources.shape != (len(audio), speaker_count) or not np.isfinite(sources).all():
        raise ValueError('Speech separation returned invalid audio. Record the group again.')
    mixture_rms = float(np.sqrt(np.mean(audio ** 2)))
    streams = []
    for index in range(speaker_count):
        source = sources[:, index]
        rms = float(np.sqrt(np.mean(source ** 2)))
        if rms < max(0.0001, mixture_rms * 0.03):
            streams.append(None)
        else:
            source = source * (0.9 / max(float(np.max(np.abs(source))), 0.0001))
            streams.append(librosa.resample(source, orig_sr=8000, target_sr=16000).astype(np.float32))
    return streams


def verify_streams(streams, code, candidates, voice_only=False):
    results = []
    for index, stream in enumerate(streams, 1):
        row = {'Stream': index, 'Transcript': '', 'Status': 'Retry', 'Reason': '',
               'student_id': None, 'roll': None, 'score': None}
        try:
            if stream is None:
                raise ValueError('No clear voice recovered from this stream.')
            if voice_only:
                sid, score, roll = verify_voice(voice_embedding_waveform(stream), candidates)
            else:
                transcript, embedding = transcribe_waveform(stream)
                row['Transcript'] = transcript
                sid, score, roll = verify_claim(transcript, code, embedding, candidates)
            row.update(student_id=sid, roll=roll, score=score, Status='Verified')
        except ValueError as exc:
            row['Reason'] = str(exc)
        results.append(row)
    # Leakage can make the same voice appear in multiple outputs. Never count it twice
    # or choose one arbitrarily: require a fresh recording for that student.
    counts = Counter(r['student_id'] for r in results if r['Status'] == 'Verified')
    for row in results:
        if row['Status'] == 'Verified' and counts[row['student_id']] > 1:
            row['Status'] = 'Retry'
            row['Reason'] = 'The same student appeared in multiple streams. Record separately.'
            row['student_id'] = None
    return results


def verify_overlapping_audio(audio_bytes, speaker_count, code, candidates):
    return verify_streams(separate_audio(audio_bytes, speaker_count), code, candidates)
