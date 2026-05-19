"""
features/image_checker.py
--------------------------
Analyses an uploaded image for signs of manipulation using:
  1. ELA  (Error Level Analysis)  — via Pillow
  2. EXIF metadata inspection      — via Pillow
  3. Block-wise noise analysis     — via OpenCV (if available)
  4. Perceptual hash               — via imagehash (if available)

Returns a manipulation score 0–100 and a base64 ELA preview image.
"""

import io
import os
import math
import base64

from PIL import Image, ImageChops, ImageEnhance

try:
    import cv2
    import numpy as np
    CV2_OK = True
except ImportError:
    CV2_OK = False

try:
    import imagehash
    HASH_OK = True
except ImportError:
    HASH_OK = False


# ── Public API ───────────────────────────────────────────────────────────────

def analyse_image(file_bytes: bytes, filename: str) -> dict:
    """
    Run all available analyses on an uploaded image.
    Returns combined result with manipulation_score, verdict, ELA preview.
    """
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
    except Exception as e:
        return {'error': f'Cannot open image: {e}'}

    # ── 1. ELA ──────────────────────────────────────────────────────────────
    ela   = _ela(img)
    ela_s = ela['score']

    # ── 2. Metadata ─────────────────────────────────────────────────────────
    meta   = _metadata(img, filename)
    meta_s = meta['score']

    # ── 3. OpenCV noise ─────────────────────────────────────────────────────
    if CV2_OK:
        cv_res = _opencv_noise(file_bytes)
        cv_s   = cv_res['score']
    else:
        cv_res = {'score': 50, 'note': 'OpenCV not installed'}
        cv_s   = 50

    # ── 4. Hash ─────────────────────────────────────────────────────────────
    hash_info = _phash(img) if HASH_OK else {'note': 'imagehash not installed'}

    # ── Overall score (weighted) ─────────────────────────────────────────────
    overall = round(ela_s * 0.45 + meta_s * 0.25 + cv_s * 0.30)
    overall = max(0, min(100, overall))

    # ── Collect human-readable issues ────────────────────────────────────────
    issues = []
    if ela_s > 65:
        issues.append('High ELA score — possible JPEG re-compression artifacts')
    if meta.get('editing_sw'):
        issues.append(f"Editing software in metadata: {meta['editing_sw']}")
    if meta.get('no_camera'):
        issues.append('No camera make/model found in EXIF')
    if meta.get('format_mismatch'):
        issues.append('File extension does not match actual image format')
    if cv_s > 55:
        issues.append('Abnormal noise distribution across image regions')
    if not issues:
        issues.append('No obvious manipulation signals detected')

    return {
        'manipulation_score': overall,
        'verdict':            _verdict(overall),
        'ela_score':          ela_s,
        'metadata_score':     meta_s,
        'opencv_score':       cv_s,
        'ela_preview':        ela['preview_b64'],
        'issues':             issues,
        'hash':               hash_info,
        'image_size':         f"{img.width} × {img.height} px",
        'image_format':       img.format or _ext_to_fmt(filename),
    }


# ── Private helpers ──────────────────────────────────────────────────────────

def _ela(img: Image.Image) -> dict:
    """
    Error Level Analysis: re-save at quality=75, measure pixel difference.
    Edited regions show disproportionately high error levels.
    """
    try:
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=75)
        buf.seek(0)
        resaved = Image.open(buf).convert('RGB')

        diff = ImageChops.difference(img, resaved)
        extrema = diff.getextrema()                       # [(min,max), ...] per channel
        max_val = max(ch[1] for ch in extrema) or 1

        # Amplify difference for visualisation
        enhanced = ImageEnhance.Brightness(diff).enhance(255.0 / max_val * 8)

        # Score = mean luminance of enhanced diff (0–100)
        data = list(enhanced.getdata())
        mean = sum(sum(px) / 3 for px in data) / len(data)
        score = min(100, round(mean / 255 * 200))

        # Encode as base64 PNG for browser preview
        out = io.BytesIO()
        enhanced.save(out, format='PNG')
        b64 = base64.b64encode(out.getvalue()).decode()

        return {'score': score, 'preview_b64': f'data:image/png;base64,{b64}'}
    except Exception as e:
        return {'score': 50, 'preview_b64': '', 'error': str(e)}


def _metadata(img: Image.Image, filename: str) -> dict:
    """Inspect EXIF for editing software fingerprints and anomalies."""
    score = 50
    result = {'editing_sw': '', 'no_camera': False, 'format_mismatch': False}

    MAKE_TAG, MODEL_TAG, SW_TAG = 271, 272, 305
    EDIT_SW = ['photoshop', 'gimp', 'lightroom', 'paint.net', 'canva',
               'snapseed', 'pixlr', 'affinity']

    try:
        exif = img._getexif() if hasattr(img, '_getexif') else None
    except Exception:
        exif = None

    if exif is None:
        score += 10
    else:
        make  = exif.get(MAKE_TAG, '')
        model = exif.get(MODEL_TAG, '')
        sw    = str(exif.get(SW_TAG, ''))

        if any(s in sw.lower() for s in EDIT_SW):
            score += 35
            result['editing_sw'] = sw

        if not make and not model:
            score += 15
            result['no_camera'] = True

    # Format vs extension mismatch
    actual = img.format or ''
    ext = os.path.splitext(filename)[1].lower()
    if ext in ('.jpg', '.jpeg') and actual not in ('JPEG', ''):
        score += 20
        result['format_mismatch'] = True

    result['score'] = max(0, min(100, score))
    return result


def _opencv_noise(file_bytes: bytes) -> dict:
    """
    Block-wise noise variance analysis.
    If one region is much noisier than average, it may have been spliced.
    """
    try:
        arr    = np.frombuffer(file_bytes, dtype=np.uint8)
        gray   = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return {'score': 50}

        h, w   = gray.shape
        bsz    = max(32, min(h, w) // 8)
        variances = [
            float(np.var(gray[y:y+bsz, x:x+bsz]))
            for y in range(0, h - bsz, bsz)
            for x in range(0, w - bsz, bsz)
        ]
        if not variances:
            return {'score': 50}

        mean_v = sum(variances) / len(variances)
        max_v  = max(variances)
        ratio  = max_v / (mean_v + 1e-6)
        score  = min(100, round(math.log1p(ratio) * 15))

        return {'score': score, 'block_ratio': round(ratio, 2)}
    except Exception as e:
        return {'score': 50, 'error': str(e)}


def _phash(img: Image.Image) -> dict:
    """Compute perceptual + difference hash for duplicate/tamper detection."""
    try:
        return {
            'phash': str(imagehash.phash(img)),
            'dhash': str(imagehash.dhash(img)),
        }
    except Exception as e:
        return {'error': str(e)}


def _verdict(score: int) -> str:
    if score >= 75: return 'Likely Manipulated'
    if score >= 50: return 'Possibly Manipulated'
    if score >= 25: return 'Likely Authentic'
    return 'Authentic'


def _ext_to_fmt(fn: str) -> str:
    return {'.jpg': 'JPEG', '.jpeg': 'JPEG', '.png': 'PNG',
            '.gif': 'GIF', '.webp': 'WEBP', '.bmp': 'BMP'
            }.get(os.path.splitext(fn)[1].lower(), 'Unknown')
