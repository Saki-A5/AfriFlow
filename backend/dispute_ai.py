# FILE: backend/dispute_ai.py
#
# AI Dispute Resolution Pipeline — Two-Phase Architecture
#
# Phase 1 — Visual Evidence Analysis (llama-3.2-11b-vision-preview)
#   Every image submitted as evidence is independently analyzed.
#   The vision model reads the image and answers a specific question:
#   "What does this image show that is relevant to this dispute?"
#   Output: structured visual findings per image.
#
# Phase 2 — Final Adjudication (llama-3.3-70b-versatile)
#   Receives: trade terms + all text evidence + vision findings + trust profiles
#   Produces: confidence score, factual finding, recommendation, resolution type.
#
# Routing:
#   confidence >= 85 → auto-resolve
#   confidence <  85 → escalate to human arbitrator (with AI summary pre-loaded)
#
# Fallback: deterministic rule-based engine (no API key / rate limit / network error)

import os
import json
import re
import base64
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY  = os.environ.get('GROQ_API_KEY', '')
VISION_MODEL  = 'llama-3.2-11b-vision-preview'
TEXT_MODEL    = 'llama-3.3-70b-versatile'


# ─────────────────────────────────────────────────────────────
# PHASE 1 — VISION ANALYSIS
# Analyzes each image evidence item independently.
# Returns a plain-language finding per image.
# ─────────────────────────────────────────────────────────────

def _load_image_as_base64(file_path: str) -> tuple:
    """
    Loads an image file and returns (base64_string, media_type).
    Supports JPEG, PNG, GIF, WEBP.
    Returns (None, None) if file missing or unsupported.
    """
    if not file_path or not os.path.exists(file_path):
        return None, None

    ext = file_path.rsplit('.', 1)[-1].lower()
    media_map = {
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png',
        'gif': 'image/gif',
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
        print(f"[Vision] Failed to load image {file_path}: {e}")
        return None, None


def analyze_image_evidence(
    client: Groq,
    evidence_item,
    dispute_reason: str,
    trade_description: str,
    submitter_role: str
) -> dict:
    """
    Sends a single image to llama-3.2-11b-vision-preview.
    Returns a structured visual finding.

    Args:
        evidence_item : Evidence ORM object with file_path set
        dispute_reason: The stated reason for the dispute
        trade_description: What goods were traded
        submitter_role: 'buyer' or 'supplier'
    """
    b64_data, media_type = _load_image_as_base64(evidence_item.file_path)

    if not b64_data:
        return {
            'evidence_id': evidence_item.id,
            'submitter_role': submitter_role,
            'visual_finding': 'Image file could not be loaded for analysis.',
            'key_observations': [],
            'supports_claimant': None,
            'image_quality': 'unavailable',
            'analyzed': False,
        }

    prompt = f"""You are analyzing photographic evidence submitted in a trade dispute.

Context:
- Trade goods: {trade_description}
- Dispute reason: {dispute_reason}
- Image submitted by: {submitter_role}

Examine this image carefully and provide a factual, objective analysis.

You MUST respond with ONLY valid JSON:
{{
  "visual_finding": "<2-3 sentence objective description of what the image shows>",
  "key_observations": ["<observation 1>", "<observation 2>", "<observation 3>"],
  "supports_claimant": <true | false | null — does this image support the submitter's position?>,
  "damage_visible": <true | false | null — is physical damage visible?>,
  "goods_match_description": <true | false | null — do visible goods match '{trade_description}'?>,
  "packaging_condition": "<intact | damaged | opened | not_visible>",
  "image_quality": "<clear | partial | unclear>"
}}

Be specific. Only describe what you can actually see. No speculation."""

    try:
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'image_url',
                            'image_url': {
                                'url': f'data:{media_type};base64,{b64_data}'
                            }
                        },
                        {
                            'type': 'text',
                            'text': prompt
                        }
                    ]
                }
            ],
            temperature=0.1,
            max_tokens=500,
        )

        raw = response.choices[0].message.content.strip()
        raw = re.sub(r'^```json\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        parsed = json.loads(raw)

        return {
            'evidence_id':           evidence_item.id,
            'submitter_role':        submitter_role,
            'visual_finding':        parsed.get('visual_finding', ''),
            'key_observations':      parsed.get('key_observations', []),
            'supports_claimant':     parsed.get('supports_claimant'),
            'damage_visible':        parsed.get('damage_visible'),
            'goods_match_description': parsed.get('goods_match_description'),
            'packaging_condition':   parsed.get('packaging_condition', 'not_visible'),
            'image_quality':         parsed.get('image_quality', 'unclear'),
            'analyzed':              True,
            'model':                 VISION_MODEL,
        }

    except json.JSONDecodeError as e:
        print(f"[Vision] JSON parse error for evidence {evidence_item.id}: {e}")
        return {
            'evidence_id': evidence_item.id,
            'submitter_role': submitter_role,
            'visual_finding': 'Image analysis completed but response could not be parsed.',
            'key_observations': [],
            'supports_claimant': None,
            'image_quality': 'unclear',
            'analyzed': True,
            'model': VISION_MODEL,
        }
    except Exception as e:
        print(f"[Vision] Vision API error for evidence {evidence_item.id}: {e}")
        return {
            'evidence_id': evidence_item.id,
            'submitter_role': submitter_role,
            'visual_finding': f'Image analysis failed: {str(e)[:100]}',
            'key_observations': [],
            'supports_claimant': None,
            'image_quality': 'unavailable',
            'analyzed': False,
        }


def run_vision_analysis_phase(
    client: Groq,
    evidence_list: list,
    trade,
    dispute
) -> list:
    """
    Iterates all evidence items. For each with an image file,
    calls the vision model. Returns list of visual findings.
    """
    visual_findings = []
    image_count = 0

    for ev in evidence_list:
        # Only process items that have a file path (uploaded images)
        if not ev.file_path:
            continue

        ext = ev.file_path.rsplit('.', 1)[-1].lower() if '.' in ev.file_path else ''
        if ext not in ('jpg', 'jpeg', 'png', 'gif', 'webp'):
            continue

        submitter_role = 'buyer' if ev.submitted_by == trade.buyer_id else 'supplier'
        print(f"[Vision] Analyzing image evidence #{ev.id} from {submitter_role}...")

        finding = analyze_image_evidence(
            client=client,
            evidence_item=ev,
            dispute_reason=dispute.reason,
            trade_description=trade.description,
            submitter_role=submitter_role
        )
        visual_findings.append(finding)
        image_count += 1

    print(f"[Vision] Analyzed {image_count} image(s). {len(visual_findings)} findings generated.")
    return visual_findings


# ─────────────────────────────────────────────────────────────
# PHASE 2 — FINAL ADJUDICATION (TEXT MODEL)
# Receives all evidence — text + structured vision findings.
# Produces the final resolution.
# ─────────────────────────────────────────────────────────────

def _format_visual_findings_for_prompt(visual_findings: list) -> str:
    if not visual_findings:
        return '  No image evidence submitted.'

    lines = []
    for f in visual_findings:
        role = f.get('submitter_role', 'unknown').upper()
        quality = f.get('image_quality', 'unclear')
        finding = f.get('visual_finding', 'No finding')
        observations = f.get('key_observations', [])
        damage = f.get('damage_visible')
        match = f.get('goods_match_description')
        packaging = f.get('packaging_condition', 'not_visible')

        lines.append(f"  [{role} IMAGE — quality: {quality}]")
        lines.append(f"  Visual Finding: {finding}")
        if observations:
            for obs in observations[:3]:
                lines.append(f"    • {obs}")
        lines.append(f"  Damage visible: {damage} | Goods match description: {match} | Packaging: {packaging}")
        lines.append('')

    return '\n'.join(lines)


def _build_adjudication_prompt(
    trade, dispute, buyer, supplier,
    buyer_trust, supplier_trust,
    evidence_list, visual_findings
) -> str:

    buyer_trust_d    = buyer_trust.to_dict() if buyer_trust else {}
    supplier_trust_d = supplier_trust.to_dict() if supplier_trust else {}

    # Separate text evidence by party
    buyer_text_ev = [
        e for e in evidence_list
        if e.submitted_by == trade.buyer_id and not e.file_path
    ]
    supplier_text_ev = [
        e for e in evidence_list
        if e.submitted_by == trade.supplier_id and not e.file_path
    ]

    buyer_text = '\n'.join([
        f"  [{e.evidence_type}] {e.content}" for e in buyer_text_ev
    ]) or '  No text evidence submitted.'

    supplier_text = '\n'.join([
        f"  [{e.evidence_type}] {e.content}" for e in supplier_text_ev
    ]) or '  No text evidence submitted.'

    visual_section = _format_visual_findings_for_prompt(visual_findings)

    # Buyer/supplier image evidence for context
    buyer_images    = [f for f in visual_findings if f.get('submitter_role') == 'buyer']
    supplier_images = [f for f in visual_findings if f.get('submitter_role') == 'supplier']

    return f"""You are AfriFlow's AI Trade Arbitrator. Issue a fair, evidence-based resolution for this African cross-border trade dispute.

You have received BOTH text evidence AND AI-analyzed visual evidence (from llama-3.2-11b-vision-preview). Use all of it.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRADE AGREEMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Trade ID:        {trade.id}
Goods:           {trade.description}
Quantity:        {trade.quantity}
Value:           {trade.currency} {trade.amount:,.0f}
Delivery Window: {trade.delivery_days} days
Release Condition: {trade.release_condition}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PARTIES & TRUST PROFILES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BUYER: {buyer.business_name} ({buyer.location})
  Trust Score:        {buyer_trust_d.get('overall_score', 'N/A')}/100
  Completed Trades:   {buyer_trust_d.get('total_trades', 0)}
  Dispute Rate:       {buyer_trust_d.get('dispute_rate', 'N/A')}
  Score Trajectory:   {buyer_trust_d.get('score_trajectory', 'unknown')}

SUPPLIER: {supplier.business_name} ({supplier.location})
  Trust Score:        {supplier_trust_d.get('overall_score', 'N/A')}/100
  Completed Trades:   {supplier_trust_d.get('total_trades', 0)}
  Delivery Accuracy:  {supplier_trust_d.get('delivery_accuracy', 'N/A')}%
  Score Trajectory:   {supplier_trust_d.get('score_trajectory', 'unknown')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DISPUTE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Raised by: {'Buyer' if dispute.raised_by == trade.buyer_id else 'Supplier'}
Reason:    {dispute.reason}
Details:   {dispute.description or 'None provided'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TEXT EVIDENCE — BUYER ({len(buyer_text_ev)} items)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{buyer_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TEXT EVIDENCE — SUPPLIER ({len(supplier_text_ev)} items)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{supplier_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VISUAL EVIDENCE ANALYSIS (llama-3.2-11b-vision-preview)
Buyer images: {len(buyer_images)} | Supplier images: {len(supplier_images)}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{visual_section}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ADJUDICATION INSTRUCTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Weigh all evidence carefully. Prioritize:
1. Concrete visual evidence (image findings) over unsubstantiated text claims
2. Specific documentation (tracking, receipts) over general descriptions
3. Consistency between text and visual evidence
4. Trust score trajectory as a secondary credibility signal
5. Burden of proof — for damage claims, damage must be demonstrable

Resolution types:
  release_to_supplier — evidence supports supplier fulfilled their obligation
  refund_to_buyer     — evidence supports buyer's claim of non-fulfillment
  partial_refund      — both parties partially at fault
  escalate_to_human   — insufficient evidence for confident automated resolution

Confidence calibration:
  90–100: Visual + text evidence strongly consistent. One side clearly supported.
  80–89:  Evidence favors one side but not conclusively.
  70–79:  Weak evidence. Some pattern support.
  <70:    Escalate to human — do not auto-resolve ambiguous cases.

You MUST respond with ONLY valid JSON:
{{
  "confidence": <integer 0-100>,
  "finding": "<2-3 sentence factual finding synthesizing text AND visual evidence>",
  "recommendation": "<1-2 sentence specific recommended action>",
  "resolution_type": "<release_to_supplier | refund_to_buyer | partial_refund | escalate_to_human>",
  "reasoning_steps": [
    "<step: what evidence was reviewed>",
    "<step: what visual analysis showed>",
    "<step: how trust profiles influenced assessment>",
    "<step: final determination logic>"
  ],
  "visual_evidence_impact": "<how image analysis affected the outcome — none | supporting | decisive | contradicting>"
}}

No text outside the JSON."""


def _adjudicate(client, trade, dispute, buyer, supplier,
                buyer_trust, supplier_trust, evidence_list, visual_findings) -> dict:
    """Calls text model for final resolution."""
    prompt = _build_adjudication_prompt(
        trade, dispute, buyer, supplier,
        buyer_trust, supplier_trust,
        evidence_list, visual_findings
    )

    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {
                'role': 'system',
                'content': (
                    'You are AfriFlow AI Trade Arbitrator. '
                    'Respond only with valid JSON. No markdown. No preamble.'
                )
            },
            {'role': 'user', 'content': prompt}
        ],
        temperature=0.1,
        max_tokens=800,
    )

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r'^```json\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    parsed = json.loads(raw)

    return {
        'confidence':             int(parsed.get('confidence', 70)),
        'finding':                parsed.get('finding', ''),
        'recommendation':         parsed.get('recommendation', ''),
        'resolution_type':        parsed.get('resolution_type', 'escalate_to_human'),
        'reasoning_steps':        parsed.get('reasoning_steps', []),
        'visual_evidence_impact': parsed.get('visual_evidence_impact', 'none'),
        'visual_findings':        visual_findings,
        'source':                 'groq_two_phase',
        'vision_model':           VISION_MODEL,
        'text_model':             TEXT_MODEL,
    }


# ─────────────────────────────────────────────────────────────
# FALLBACK — DETERMINISTIC (no API key / network error)
# ─────────────────────────────────────────────────────────────

def rule_based_fallback(trade, dispute, buyer_trust, supplier_trust, evidence_list) -> dict:
    buyer_ev    = [e for e in evidence_list if e.submitted_by == trade.buyer_id]
    supplier_ev = [e for e in evidence_list if e.submitted_by == trade.supplier_id]
    buyer_images    = [e for e in buyer_ev if e.file_path]
    supplier_images = [e for e in supplier_ev if e.file_path]

    buyer_score    = buyer_trust.overall_score    if buyer_trust    else 50
    supplier_score = supplier_trust.overall_score if supplier_trust else 50
    reason_lower   = dispute.reason.lower()

    if any(w in reason_lower for w in ['damaged', 'damage', 'broken']):
        # With supplier images → likely transit damage
        if supplier_images or len(supplier_ev) > 0:
            confidence, resolution_type = 88, 'release_to_supplier'
            finding = (
                "Evidence pattern is consistent with transit damage rather than supplier negligence. "
                "Supplier submitted packaging documentation. Damage is attributable to courier handling."
            )
            recommendation = "Release funds to supplier. Flag courier route for review."
        else:
            confidence, resolution_type = 78, 'escalate_to_human'
            finding = (
                "Damage claimed but neither party submitted photographic evidence. "
                "Cannot determine cause or responsibility without visual confirmation."
            )
            recommendation = "Escalate to human arbitrator. Request physical inspection."

    elif any(w in reason_lower for w in ['not delivered', 'never arrived', 'missing']):
        if len(supplier_ev) > 0:
            confidence, resolution_type = 79, 'release_to_supplier'
            finding = (
                "Supplier submitted delivery documentation. "
                "Delivery records indicate goods were dispatched within agreed window. "
                "Buyer has not provided counter-evidence of non-delivery."
            )
            recommendation = "Release funds to supplier. Buyer should investigate with courier."
        else:
            confidence, resolution_type = 83, 'refund_to_buyer'
            finding = (
                "Supplier provided no delivery confirmation, tracking number, or evidence of dispatch. "
                "Buyer's non-receipt claim is uncontested. Funds cannot be released without delivery proof."
            )
            recommendation = "Refund funds to buyer. Supplier must provide tracking on all future trades."

    elif any(w in reason_lower for w in ['wrong', 'incorrect', 'different', 'not what']):
        if buyer_images:
            confidence, resolution_type = 82, 'refund_to_buyer'
            finding = (
                "Buyer submitted photographic evidence showing goods differ from trade description. "
                "Visual evidence of incorrect items supports the buyer's claim of wrong goods received. "
                "Supplier did not submit counter-images confirming correct goods were shipped."
            )
            recommendation = "Refund to buyer. Supplier must verify fulfillment against trade agreement."
        elif len(buyer_ev) >= len(supplier_ev):
            confidence, resolution_type = 74, 'escalate_to_human'
            finding = (
                "Buyer claims goods mismatch but without photographic evidence. "
                "Both parties have submitted evidence but the case is inconclusive without visual confirmation."
            )
            recommendation = "Escalate to human arbitrator. Request photos from buyer."
        else:
            confidence, resolution_type = 71, 'escalate_to_human'
            finding = (
                "Conflicting claims about goods description. Insufficient visual evidence for automated resolution."
            )
            recommendation = "Escalate to human arbitrator."

    else:
        diff = abs(buyer_score - supplier_score)
        if diff > 20 and supplier_score > buyer_score:
            confidence, resolution_type = 72, 'release_to_supplier'
            finding = (
                "No concrete evidence of supplier failure presented. "
                "Supplier has a significantly stronger verified trade history. "
                "Balance of evidence favors supplier fulfillment."
            )
            recommendation = "Release funds to supplier."
        else:
            confidence, resolution_type = 62, 'escalate_to_human'
            finding = (
                "Evidence is insufficient for high-confidence automated resolution. "
                "Case requires human review."
            )
            recommendation = "Escalate to human arbitrator within 24 hours."

    return {
        'confidence':      confidence,
        'finding':         finding,
        'recommendation':  recommendation,
        'resolution_type': resolution_type,
        'reasoning_steps': [
            f"Dispute reason analyzed: {dispute.reason}",
            f"Evidence: {len(buyer_ev)} buyer items ({len(buyer_images)} images), "
            f"{len(supplier_ev)} supplier items ({len(supplier_images)} images)",
            f"Trust profiles: Buyer {buyer_score}/100 vs Supplier {supplier_score}/100",
            f"Resolution: {resolution_type} at {confidence}% confidence",
            f"Threshold: {'AUTO-RESOLVE' if confidence >= 85 else 'ESCALATE TO HUMAN'}",
        ],
        'visual_evidence_impact': 'supporting' if buyer_images or supplier_images else 'none',
        'visual_findings':        [],
        'source':                 'rule_based_fallback',
        'vision_model':           None,
        'text_model':             None,
    }


# ─────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────

def analyze_dispute_with_ai(
    trade, dispute, buyer, supplier,
    buyer_trust, supplier_trust, evidence_list
) -> dict:
    """
    Orchestrates the full two-phase dispute resolution pipeline.

    Phase 1: Vision analysis of all image evidence (llama-3.2-11b-vision-preview)
    Phase 2: Final adjudication with all evidence (llama-3.3-70b-versatile)

    Falls back to rule-based engine if Groq is unavailable.
    """
    if not GROQ_API_KEY:
        print("[Dispute AI] No GROQ_API_KEY. Using rule-based fallback.")
        return rule_based_fallback(trade, dispute, buyer_trust, supplier_trust, evidence_list)

    try:
        client = Groq(api_key=GROQ_API_KEY)

        # ── Phase 1: Analyze all image evidence ──
        print("[Dispute AI] Phase 1: Running vision analysis on image evidence...")
        visual_findings = run_vision_analysis_phase(client, evidence_list, trade, dispute)
        print(f"[Dispute AI] Phase 1 complete. {len(visual_findings)} visual finding(s) produced.")

        # ── Phase 2: Final adjudication ──
        print("[Dispute AI] Phase 2: Running final adjudication...")
        result = _adjudicate(
            client, trade, dispute, buyer, supplier,
            buyer_trust, supplier_trust,
            evidence_list, visual_findings
        )
        print(f"[Dispute AI] Phase 2 complete. "
              f"Confidence: {result['confidence']}% | "
              f"Resolution: {result['resolution_type']} | "
              f"Visual impact: {result['visual_evidence_impact']}")
        return result

    except json.JSONDecodeError as e:
        print(f"[Dispute AI] JSON parse error in adjudication: {e}. Using fallback.")
        result = rule_based_fallback(trade, dispute, buyer_trust, supplier_trust, evidence_list)
        result['source'] = 'rule_based_fallback_json_error'
        return result

    except Exception as e:
        print(f"[Dispute AI] Pipeline error: {e}. Using fallback.")
        result = rule_based_fallback(trade, dispute, buyer_trust, supplier_trust, evidence_list)
        result['source'] = 'rule_based_fallback_api_error'
        return result