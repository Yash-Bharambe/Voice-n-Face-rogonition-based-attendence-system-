from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np 
import io
import librosa
import streamlit as st
from pathlib import Path
from threading import Lock
from src.pipelines.voice_audio import prepare_audio

MODEL_DIRECTORY = Path(__file__).resolve().parents[2] / '.local' / 'models'
TRANSCRIPTION_LOCK = Lock()

@st.cache_resource
def load_transcriber():
    from faster_whisper import WhisperModel
    local = MODEL_DIRECTORY / 'whisper-small-en'
    if not (local / 'model.bin').is_file():
        snapshots = MODEL_DIRECTORY / 'models--Systran--faster-whisper-small.en' / 'snapshots'
        cached = sorted(snapshots.glob('*/model.bin'))
        if cached:
            local = cached[-1].parent
    model = str(local) if (local / 'model.bin').is_file() else 'small.en'
    return WhisperModel(model, device='cpu', compute_type='int8', download_root=str(MODEL_DIRECTORY), cpu_threads=4)

def transcribe_claim(audio_bytes):
    audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000)
    return transcribe_waveform(audio)

def voice_embedding_waveform(audio):
    if not 3 <= len(audio) / 16000 <= 20:
        raise ValueError('Record 3–20 seconds of clear speech.')
    audio, _ = prepare_audio(audio)
    wav = preprocess_wav(audio)
    if len(wav) / 16000 < 2:
        raise ValueError('Too little clear speech. Speak for longer with fewer pauses.')
    return load_voice_encoder().embed_utterance(wav)

def voice_embedding_bytes(audio_bytes):
    audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000)
    return voice_embedding_waveform(audio)

def transcribe_waveform(audio):
    if not 3 <= len(audio) / 16000 <= 20:
        raise ValueError('Record one student speaking for 3–20 seconds.')
    audio, _ = prepare_audio(audio)
    # Never prompt ASR with the expected code or roll; that could bias recognition.
    with TRANSCRIPTION_LOCK:
        segments, _ = load_transcriber().transcribe(audio, language='en', beam_size=5,
            condition_on_previous_text=False, word_timestamps=True)
        segments = list(segments)
    if not segments or any(s.no_speech_prob > 0.6 or s.avg_logprob < -1.0 for s in segments):
        raise ValueError('Speech was unclear. Repeat the complete phrase.')
    words = [w for s in segments for w in (s.words or [])]
    if not words or any(w.probability < 0.2 for w in words) or np.mean([w.probability for w in words]) < 0.65:
        raise ValueError('Some words were unclear. Repeat the complete phrase.')
    transcript = ' '.join(s.text.strip() for s in segments)
    voice_audio = preprocess_wav(audio)
    if len(voice_audio)/16000 < 2:
        raise ValueError('Too little clear speech for voice matching. Speak the complete phrase with fewer pauses.')
    embedding = load_voice_encoder().embed_utterance(voice_audio)
    return transcript, embedding


@st.cache_resource
def load_voice_encoder():
    return VoiceEncoder()


def get_voice_embedding(audio_bytes):
    try:
        encoder = load_voice_encoder()

        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        wav = preprocess_wav(audio)
        embedding = encoder.embed_utterance(wav)
        return embedding.tolist()
    except Exception as e:
        st.error('Voice recog error')
        return None

def enroll_voice_sample(audio_bytes):
    if not audio_bytes:
        raise ValueError('A voice recording is required.')
    try:
        audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000)
    except Exception as exc:
        raise ValueError('Could not read the voice recording. Record again.') from exc
    if not 8 <= len(audio)/16000 <= 30:
        raise ValueError('Record 8–30 seconds of your own voice.')
    audio, _ = prepare_audio(audio)
    wav = preprocess_wav(audio)
    if len(wav)/16000 < 4:
        raise ValueError('Record at least four seconds of clear speech, with fewer pauses.')
    embedding = np.asarray(load_voice_encoder().embed_utterance(wav))
    if embedding.shape != (256,) or not np.isfinite(embedding).all() or np.linalg.norm(embedding) == 0:
        raise ValueError('Voice enrollment failed. Record again.')
    return embedding.tolist()
    

def identify_speaker(new_embedding, candidates_dict, threshold=0.65):
    if new_embedding is None or not candidates_dict:
        return None, 0.0
    
    best_sid = None
    best_score = -1.0

    for sid, stored_embedding in candidates_dict.items():
        if stored_embedding:
            similarity = np.dot(new_embedding, stored_embedding)
            if similarity> best_score:
                best_score = similarity
                best_sid = sid

    if best_score >= threshold:
        return best_sid, best_score
    
    return None, best_score



def process_bulk_audio(audio_bytes, candidates_dict, threshold=0.65):

    try:
        encoder = load_voice_encoder()

        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        segments = librosa.effects.split(audio, top_db=30)

        identified_results = {}


        for start, end in segments:

            if (end-start) < sr * 0.5:
                continue
            segment_audio = audio[start:end]
            wav = preprocess_wav(segment_audio)
            embedding = encoder.embed_utterance(wav)


            sid, score = identify_speaker(embedding, candidates_dict, threshold)

            if sid:
                if sid not in identified_results or score > identified_results[sid]:
                    identified_results[sid] = score

        return identified_results
    except Exception as e:
        st.error('Bulk process error')
        return {}
