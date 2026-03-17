# FILE: backend/id_verification.py
#
# AI Image Verification Module — llama-3.2-11b-vision-preview
#
# Two verification jobs:
#
# Job 1 — ID Document Verification (called during registration)
#   Checks: is this a real government ID? Does it match the declared type?
#   Is there a visible name? Any signs of tampering or digital editing?
#   Output: confidence score, name extracted, flags, pass/fail decision.
#
# Job 2 — General Image Analysis (called for any uploaded image)
#   Checks: what is in this image? Is it relevant to the claimed context?
#   Is it legible? Is it a real photograph or a screenshot/edited image?
#   Output: description, authenticity assessment, relevance to context.
#
# Both use the same vision model but different prompts and output schemas.

import os
import re
import json
import base64
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY  = os.environ.get('GROQ_API_KEY', '')
VISION_MODEL  = 'llama-3.2-11b-vision-preview'

ALLOWED_IMAGE_EXTS = {'jpg', 'jpeg', 'png', 'gif', 'webp'}
ALLOWED_DOC_EXTS   = {'jpg', 'jpeg', 'png', 'gif', 'webp', 'pdf'}

# Confidence thresholds
ID_PASS_THRESHOLD   = 70   # Minimum confidence to accept an ID as valid
ID_REVIEW_THRESHOLD = 50   # Below this → automatic rejection


# ─────────────────────────────────────────────────────────────
# SHARED UTILITY
# ─────────────────────────────────────────────────────────────

def load_image_base64(file_path: str) -> tuple:
    """
    Loads an image file and returns (base64_string, media_type).
    Returns (None, None) on failure.
    """
    if not file_path or not os.path.exists(file_path):
        return None, None

    ext = file_path.rsplit('.', 1)[-1].lower() if '.' in file_path else ''
    media_map = {
        'jpg': 'image/jpeg', 'jpeg': 'image/jpeg',
        'png': 'image/png',  'gif': 'image/gif',
        'webp': 'image/webp',
    }
    media_type = media_map.get(ext)
    if not media_type:
        return None, None

    try:
        with open(file_path, 'rb') as f:
            b64 = base64.standard_b64encode(f.read()).decode('utf-8')
        return b64, media_type
    except Exception as e:
        print(f"[IDVerify] Failed to load image {file_path}: {e}")
        return None, None


def _call_vision(client: Groq, b64_data: str, media_type: str, prompt: str, max_tokens: int = 600) -> dict:
    """Sends a base64 image to the vision model and parses JSON response."""
    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'image_url',
                        'image_url': {'url': f'data:{media_type};base64,{b64_data}'}
                    },
                    {'type': 'text', 'text': prompt}
                ]
            }
        ],
        temperature=0.05,   # Near-deterministic for verification
        max_tokens=max_tokens,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r'^```json\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    return json.loads(raw)


# ─────────────────────────────────────────────────────────────
# JOB 1 — ID DOCUMENT VERIFICATION
# ─────────────────────────────────────────────────────────────

ID_TYPE_LABELS = {
    'passport':         'international passport',
    'national_id':      'national identity card',
    'drivers_license':  "driver's licence",
    'voters_card':      "voter's card / PVC",
}

def _build_id_verification_prompt(declared_type: str, owner_name: str) -> str:
    declared_label = ID_TYPE_LABELS.get(declared_type, declared_type)
    return f"""You are AfriFlow's identity verification AI. Examine this image carefully.

The user claims this is a {declared_label} belonging to: {owner_name}

Your job is to verify:
1. Is this actually a government-issued identity document?
2. Does the document type match the declared type ({declared_label})?
3. Is there a visible name on the document? What does it say?
4. Is the document image clear enough to read?
5. Are there any signs of digital tampering, photo editing, or this being a screenshot of another photo?
6. Is the document from an African country (Nigeria, Ghana, Kenya, South Africa, etc.)?

Be strict but fair. A slightly blurry photo of a real ID is acceptable.
A clearly fake document, a stock photo, a screenshot of someone else's ID, or completely unreadable image is not.

You MUST respond with ONLY valid JSON:
{{
  "is_identity_document": <true | false — is this clearly an ID document of any kind?>,
  "declared_type_matches": <true | false | null — does it match '{declared_label}'? null if unclear>,
  "name_visible": <true | false>,
  "name_on_document": "<visible name text, or null if not readable>",
  "image_quality": "<clear | acceptable | poor | unusable>",
  "appears_authentic": <true | false — no obvious digital editing or tampering>,
  "is_screenshot_or_copy": <true | false — looks like a photo of a screen or photocopy of a photocopy>,
  "country_detected": "<country name if identifiable, or null>",
  "confidence": <integer 0-100 — confidence this is a real, matching, readable ID>,
  "rejection_reason": "<null if passing, or specific reason for rejection>",
  "flags": ["<flag1>", "<flag2>"] 
}}

Possible flags: low_quality, type_mismatch, name_mismatch, possible_tampering, screenshot_detected, not_an_id, unusable_image, wrong_country

No text outside the JSON."""


def verify_id_document(file_path: str, declared_type: str, owner_name: str) -> dict:
    """
    Sends an ID document image to llama-3.2-11b-vision-preview for verification.

    Args:
        file_path:      Path to the uploaded ID image
        declared_type:  One of: passport | national_id | drivers_license | voters_card
        owner_name:     The name the user registered with

    Returns:
        dict with keys: passed, confidence, name_on_document, flags,
                        rejection_reason, image_quality, appears_authentic,
                        declared_type_matches, source
    """
    # ── No API key — fallback ─────────────────
    if not GROQ_API_KEY:
        print("[IDVerify] No GROQ_API_KEY. Using acceptance fallback.")
        return _id_fallback(file_path, declared_type, 'no_api_key')

    b64_data, media_type = load_image_base64(file_path)

    # ── File not loadable ─────────────────────
    if not b64_data:
        return {
            'passed':               False,
            'confidence':           0,
            'name_on_document':     None,
            'flags':                ['unusable_image'],
            'rejection_reason':     'The uploaded file could not be read. Please upload a clear photo (JPEG or PNG).',
            'image_quality':        'unusable',
            'appears_authentic':    None,
            'declared_type_matches': None,
            'is_identity_document': False,
            'source':               'load_error',
        }

    prompt = _build_id_verification_prompt(declared_type, owner_name)

    try:
        client = Groq(api_key=GROQ_API_KEY)
        parsed = _call_vision(client, b64_data, media_type, prompt, max_tokens=600)

        confidence   = int(parsed.get('confidence', 0))
        flags        = parsed.get('flags', [])
        is_id        = parsed.get('is_identity_document', False)
        authentic    = parsed.get('appears_authentic', True)
        quality      = parsed.get('image_quality', 'poor')
        is_screenshot = parsed.get('is_screenshot_or_copy', False)

        # Add computed flags
        if is_screenshot and 'screenshot_detected' not in flags:
            flags.append('screenshot_detected')
        if quality == 'unusable' and 'unusable_image' not in flags:
            flags.append('unusable_image')
        if not is_id and 'not_an_id' not in flags:
            flags.append('not_an_id')

        # Determine pass/fail
        if not is_id:
            passed = False
            rejection_reason = parsed.get('rejection_reason') or \
                'This does not appear to be a government-issued identity document.'
        elif quality == 'unusable':
            passed = False
            rejection_reason = 'The image is too blurry or dark to read. Please upload a clearer photo.'
        elif not authentic or is_screenshot:
            passed = False
            rejection_reason = parsed.get('rejection_reason') or \
                'The document appears to have been digitally edited or is a screenshot. Please upload a direct photo of your ID.'
        elif confidence < ID_REVIEW_THRESHOLD:
            passed = False
            rejection_reason = parsed.get('rejection_reason') or \
                f'Could not verify this document with sufficient confidence ({confidence}%). Please upload a clearer photo.'
        else:
            passed = True
            rejection_reason = None

        print(f"[IDVerify] Result: passed={passed} confidence={confidence} flags={flags}")

        return {
            'passed':               passed,
            'confidence':           confidence,
            'name_on_document':     parsed.get('name_on_document'),
            'flags':                flags,
            'rejection_reason':     rejection_reason,
            'image_quality':        quality,
            'appears_authentic':    authentic,
            'declared_type_matches': parsed.get('declared_type_matches'),
            'is_identity_document': is_id,
            'country_detected':     parsed.get('country_detected'),
            'source':               'groq_vision',
            'model':                VISION_MODEL,
        }

    except json.JSONDecodeError as e:
        print(f"[IDVerify] JSON parse error: {e}")
        return _id_fallback(file_path, declared_type, 'json_error')

    except Exception as e:
        print(f"[IDVerify] Vision API error: {e}")
        return _id_fallback(file_path, declared_type, 'api_error')


def _id_fallback(file_path: str, declared_type: str, reason: str) -> dict:
    """
    Fallback when vision API is unavailable.
    Accepts the document if the file exists and is a supported format.
    In production, this would queue for manual review.
    """
    exists = bool(file_path and os.path.exists(file_path))
    ext    = file_path.rsplit('.', 1)[-1].lower() if file_path and '.' in file_path else ''
    valid_format = ext in ALLOWED_IMAGE_EXTS

    return {
        'passed':               exists and valid_format,
        'confidence':           60 if (exists and valid_format) else 0,
        'name_on_document':     None,
        'flags':                ['ai_unavailable'],
        'rejection_reason':     None if (exists and valid_format) else 'Invalid file format.',
        'image_quality':        'unknown',
        'appears_authentic':    None,
        'declared_type_matches': None,
        'is_identity_document': None,
        'country_detected':     None,
        'source':               f'fallback_{reason}',
        'model':                None,
    }


# ─────────────────────────────────────────────────────────────
# JOB 2 — GENERAL IMAGE ANALYSIS
# Analyzes any image for relevance, authenticity, and content.
# Used for: dispute evidence, shipment photos, delivery proofs.
# ─────────────────────────────────────────────────────────────

def _build_general_analysis_prompt(context: str, purpose: str) -> str:
    return f"""You are AfriFlow's image analysis AI. Examine this image carefully.

Context: {context}
Purpose of this image: {purpose}

Analyze the image and provide an objective assessment.

You MUST respond with ONLY valid JSON:
{{
  "description": "<2-3 sentence factual description of what this image contains>",
  "is_real_photograph": <true | false — is this a real photo, not a screenshot/stock image/AI-generated?>,
  "is_relevant_to_context": <true | false | null — does this image relate to the stated context?>,
  "key_details": ["<visible detail 1>", "<visible detail 2>", "<visible detail 3>"],
  "image_quality": "<clear | acceptable | poor | unusable>",
  "contains_text": <true | false — is there readable text visible in the image?>,
  "visible_text": "<any readable text found in image, or null>",
  "potential_issues": ["<issue 1 if any>"],
  "authenticity_confidence": <integer 0-100 — confidence this is a genuine unedited photo>,
  "relevance_confidence": <integer 0-100 — confidence this image is relevant to the stated context>
}}

Be objective and specific. Only report what you can actually see.
No text outside the JSON."""


def analyze_general_image(file_path: str, context: str, purpose: str) -> dict:
    """
    Analyzes any uploaded image for authenticity, content, and relevance.

    Args:
        file_path: Path to the uploaded image
        context:   What this image is related to (e.g., "a trade dispute over Ankara fabric")
        purpose:   What the image is supposed to show (e.g., "delivery proof from supplier")

    Returns:
        dict with description, authenticity_confidence, relevance_confidence,
             key_details, visible_text, potential_issues, image_quality
    """
    if not GROQ_API_KEY:
        return _general_fallback(file_path, 'no_api_key')

    b64_data, media_type = load_image_base64(file_path)

    if not b64_data:
        return {
            'description':            'Image could not be loaded for analysis.',
            'is_real_photograph':     None,
            'is_relevant_to_context': None,
            'key_details':            [],
            'image_quality':          'unusable',
            'contains_text':          False,
            'visible_text':           None,
            'potential_issues':       ['file_unreadable'],
            'authenticity_confidence': 0,
            'relevance_confidence':    0,
            'source':                 'load_error',
        }

    prompt = _build_general_analysis_prompt(context, purpose)

    try:
        client = Groq(api_key=GROQ_API_KEY)
        parsed = _call_vision(client, b64_data, media_type, prompt, max_tokens=500)

        print(f"[ImageAnalysis] quality={parsed.get('image_quality')} "
              f"authentic={parsed.get('authenticity_confidence')} "
              f"relevant={parsed.get('relevance_confidence')}")

        return {
            'description':            parsed.get('description', ''),
            'is_real_photograph':     parsed.get('is_real_photograph'),
            'is_relevant_to_context': parsed.get('is_relevant_to_context'),
            'key_details':            parsed.get('key_details', []),
            'image_quality':          parsed.get('image_quality', 'unknown'),
            'contains_text':          parsed.get('contains_text', False),
            'visible_text':           parsed.get('visible_text'),
            'potential_issues':       parsed.get('potential_issues', []),
            'authenticity_confidence': int(parsed.get('authenticity_confidence', 50)),
            'relevance_confidence':    int(parsed.get('relevance_confidence', 50)),
            'source':                 'groq_vision',
            'model':                  VISION_MODEL,
        }

    except json.JSONDecodeError as e:
        print(f"[ImageAnalysis] JSON parse error: {e}")
        return _general_fallback(file_path, 'json_error')

    except Exception as e:
        print(f"[ImageAnalysis] Vision API error: {e}")
        return _general_fallback(file_path, 'api_error')


def _general_fallback(file_path: str, reason: str) -> dict:
    exists = bool(file_path and os.path.exists(file_path))
    return {
        'description':            'Image uploaded. AI analysis temporarily unavailable.' if exists else 'File not found.',
        'is_real_photograph':     None,
        'is_relevant_to_context': None,
        'key_details':            [],
        'image_quality':          'unknown',
        'contains_text':          False,
        'visible_text':           None,
        'potential_issues':       ['ai_unavailable'],
        'authenticity_confidence': 60 if exists else 0,
        'relevance_confidence':    60 if exists else 0,
        'source':                 f'fallback_{reason}',
        'model':                  None,
    }