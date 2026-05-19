"""
features/voice_checker.py
--------------------------
Converts an uploaded audio file to text, then feeds the transcript
through the existing Fake News ML model.

Supported formats:
  - .wav  (no extra deps needed)
  - .mp3 / .ogg / .flac / .m4a  (requires pydub + ffmpeg installed)

Speech recognition: Google STT via SpeechRecognition (free, no API key).
"""

import io
import os

# ── Optional imports ─────────────────────────────────────────────────────────
try:
    import speech_recognition as sr
    SR_OK = True
except ImportError:
    SR_OK = False

try:
    from pydub import AudioSegment
    PYDUB_OK = True
except ImportError:
    PYDUB_OK = False

SUPPORTED = {'.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aiff'}


# ── Public API ───────────────────────────────────────────────────────────────

def transcribe_audio(file_bytes: bytes, filename: str) -> dict:
    """
    Convert audio file to text transcript.
    Returns: { transcript, word_count, error? }
    """
    if not SR_OK:
        return {'error': 'SpeechRecognition not installed. Run: pip install SpeechRecognition'}

    ext = os.path.splitext(filename)[1].lower()
    if ext not in SUPPORTED:
        return {'error': f'Unsupported format: {ext}. Accepted: {", ".join(sorted(SUPPORTED))}'}

    # Convert non-WAV to WAV
    wav_bytes = _to_wav(file_bytes, ext)
    if isinstance(wav_bytes, dict):          # error dict returned
        return wav_bytes

    # Transcribe
    return _recognise(wav_bytes)


# ── Private helpers ──────────────────────────────────────────────────────────

def _to_wav(file_bytes: bytes, ext: str) -> bytes | dict:
    """Convert any supported format to WAV bytes. WAV passes through directly."""
    if ext == '.wav':
        return file_bytes

    if not PYDUB_OK:
        return {
            'error': (
                f'pydub is required to process {ext} files. '
                'Install it with: pip install pydub  '
                '(and install ffmpeg on your system for MP3/OGG support). '
                'Alternatively upload a .wav file.'
            )
        }

    try:
        fmt = ext.lstrip('.')
        seg = AudioSegment.from_file(io.BytesIO(file_bytes), format=fmt)
        seg = seg.set_channels(1).set_frame_rate(16000)   # mono 16 kHz
        buf = io.BytesIO()
        seg.export(buf, format='wav')
        return buf.getvalue()
    except Exception as e:
        return {
            'error': (
                f'Audio conversion failed: {e}. '
                'Make sure ffmpeg is installed on your system.'
            )
        }


def _recognise(wav_bytes: bytes) -> dict:
    """Run Google STT on WAV bytes. Returns transcript dict."""
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300

    try:
        with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.record(source)

        transcript = recognizer.recognize_google(audio, language='en-US')
        return {
            'transcript': transcript,
            'word_count': len(transcript.split()),
        }

    except sr.UnknownValueError:
        return {'error': 'Speech not recognised. The audio may be unclear or contain no speech.'}
    except sr.RequestError as e:
        return {'error': f'Google STT service unavailable: {e}. Check your internet connection.'}
    except Exception as e:
        return {'error': f'Transcription error: {e}'}
