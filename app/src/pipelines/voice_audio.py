import numpy as np

def audio_quality(audio):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        raise ValueError('The recording is empty or invalid. Record again.')
    centered = audio - np.mean(audio)
    rms = float(np.sqrt(np.mean(centered ** 2)))
    return {'rms': rms, 'dbfs': float(20*np.log10(max(rms,1e-9))),
            'clipping': float(np.mean(np.abs(audio) >= 0.99))}

def prepare_audio(audio):
    audio = np.asarray(audio, dtype=np.float32)
    quality = audio_quality(audio)
    if quality['rms'] < 1e-5:
        raise ValueError('No usable audio signal was recorded. Check your microphone.')
    if quality['clipping'] > 0.05:
        raise ValueError('The microphone is clipping. Move farther away or lower its recording volume.')
    centered = audio - np.mean(audio)
    # At most 20 dB of gain. Never claim to remove noise or reconstruct clipped audio.
    gain = min(10., 0.1/quality['rms'], 0.98/max(float(np.max(np.abs(centered))),1e-9))
    return np.ascontiguousarray(centered*gain, dtype=np.float32), quality
