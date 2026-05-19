"""
features/lang_support.py
-------------------------
Multi-language fake news detection.

Workflow:
  1. langdetect  → detect language of input text
  2. deep-translator → translate Hindi / Marathi → English (if needed)
  3. Feed English text to the existing ML model

Supported languages: English (en), Hindi (hi), Marathi (mr)
"""

# ── Optional imports ─────────────────────────────────────────────────────────
try:
    from langdetect import detect, LangDetectException
    LANGDETECT_OK = True
except ImportError:
    LANGDETECT_OK = False

try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_OK = True
except ImportError:
    TRANSLATOR_OK = False

# Human-readable language names
LANG_NAMES = {
    'en': 'English',
    'hi': 'Hindi',
    'mr': 'Marathi',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'ar': 'Arabic',
    'zh-cn': 'Chinese (Simplified)',
    'pt': 'Portuguese',
    'ru': 'Russian',
}

# Languages that need translation before prediction
NEEDS_TRANSLATION = {'hi', 'mr'}


# ── Public API ───────────────────────────────────────────────────────────────

def detect_language(text: str) -> dict:
    """
    Detect the language of a text string.
    Returns { lang_code, lang_name, confidence_note }
    """
    if not LANGDETECT_OK:
        return {
            'lang_code': 'en',
            'lang_name': 'English (fallback — langdetect not installed)',
        }

    if len(text.strip()) < 10:
        return {'error': 'Text too short for reliable language detection (min 10 chars).'}

    try:
        code = detect(text)
        return {
            'lang_code': code,
            'lang_name': LANG_NAMES.get(code, f'Unknown ({code})'),
        }
    except LangDetectException:
        return {'lang_code': 'en', 'lang_name': 'Unknown (defaulting to English)'}
    except Exception as e:
        return {'error': f'Language detection failed: {e}'}


def translate_to_english(text: str, source_lang: str) -> dict:
    """
    Translate text from source_lang to English.
    Returns { translated_text } or { error }.
    """
    if source_lang == 'en':
        return {'translated_text': text, 'note': 'Already English — no translation needed.'}

    if not TRANSLATOR_OK:
        return {
            'error': (
                'deep-translator not installed. '
                'Run: pip install deep-translator  '
                'Translation skipped — using original text for prediction.'
            ),
            'translated_text': text,   # use original as fallback
            'fallback': True,
        }

    try:
        # deep-translator uses ISO codes; map 'mr' → auto if needed
        translated = GoogleTranslator(source=source_lang, target='en').translate(text)
        if not translated:
            raise ValueError('Empty translation returned')
        return {'translated_text': translated}
    except Exception as e:
        # Fallback: return original so prediction can still run
        return {
            'error': f'Translation error: {e}. Using original text for prediction.',
            'translated_text': text,
            'fallback': True,
        }


def prepare_for_prediction(text: str) -> dict:
    """
    Full pipeline: detect language → translate if needed → return English text.
    Returns:
      {
        original_text,
        detected_lang_code,
        detected_lang_name,
        translated_text,        ← always present (may equal original)
        translation_note?,
        error?,
      }
    """
    result = {'original_text': text}

    # Step 1: detect
    lang_info = detect_language(text)
    if 'error' in lang_info:
        # Cannot detect; assume English
        result.update({
            'detected_lang_code': 'en',
            'detected_lang_name': 'Unknown',
            'translated_text': text,
            'translation_note': lang_info['error'],
        })
        return result

    result['detected_lang_code'] = lang_info['lang_code']
    result['detected_lang_name'] = lang_info['lang_name']

    # Step 2: translate if needed
    if lang_info['lang_code'] in NEEDS_TRANSLATION:
        trans = translate_to_english(text, lang_info['lang_code'])
        result['translated_text'] = trans.get('translated_text', text)
        if 'note' in trans:
            result['translation_note'] = trans['note']
        if 'error' in trans:
            result['translation_warning'] = trans['error']
        if trans.get('fallback'):
            result['translation_note'] = 'Translation failed; using original text.'
    else:
        result['translated_text'] = text
        if lang_info['lang_code'] != 'en':
            result['translation_note'] = (
                f"{lang_info['lang_name']} detected. "
                'Direct prediction on non-English text (model is English-trained).'
            )

    return result
