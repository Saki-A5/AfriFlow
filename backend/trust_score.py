# FILE: backend/trust_score.py
#
# AI-Powered Trust Score Engine
#
# Architecture:
#   Layer 1 — compute_signals()      : Feature engineering from raw trade events
#   Layer 2 — score_from_signals_ai(): Groq LLM interprets patterns → predictive score
#   Layer 3 — score_from_signals_fb(): Deterministic fallback (no API key / rate limit)
#   Layer 4 — calculate_and_update() : Persistence to TrustScore table
#
# The AI model (Layer 2) does what a formula cannot:
# it reads the PATTERN of signals and predicts future reliability,
# not just what happened — but what this business will do next.

import os
import json
import re
import datetime
from groq import Groq
from models import db, Trade, TrustScore, Dispute

GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
TEXT_MODEL   = 'llama-3.3-70b-versatile'
SCORE_MIN    = 10.0
SCORE_MAX    = 100.0


# ─────────────────────────────────────────────────────────────
# LAYER 1 — FEATURE ENGINEERING
# Extracts normalized behavioral signals from raw trade events.
# This layer is model-agnostic: same signals feed AI or fallback.
# ─────────────────────────────────────────────────────────────

def compute_signals(user_id: int) -> dict:
    """
    Transforms raw trade event history into a structured behavioral
    signal profile. Every signal maps to a specific behavioral question.

    Signal definitions:
      payment_reliability  — Does this buyer fund escrow promptly?
      delivery_accuracy    — Does this supplier deliver as described?
      dispute_rate         — How often is this business in disputes?
      dispute_win_rate     — When disputed, does evidence favor them?
      corridor_experience  — Are they consistent across trade routes?
      volume_score         — Are they actively and consistently trading?
      trade_velocity       — Recent trade frequency (recency signal)
      clean_streak         — Consecutive clean trades without dispute
    """

    settled = Trade.query.filter(
        (Trade.buyer_id == user_id) | (Trade.supplier_id == user_id)
    ).filter(Trade.status.in_(['settled', 'refunded'])).order_by(Trade.updated_at.desc()).all()

    all_trades = Trade.query.filter(
        (Trade.buyer_id == user_id) | (Trade.supplier_id == user_id)
    ).order_by(Trade.updated_at.desc()).all()

    disputes = Dispute.query.join(Trade, Dispute.trade_id == Trade.id).filter(
        (Trade.buyer_id == user_id) | (Trade.supplier_id == user_id)
    ).all()

    total_trades    = len(settled)
    total_disputes  = len(disputes)
    all_trade_count = len(all_trades)

    # ── Disputes won (AI resolved in this user's favor) ──
    disputes_won = 0
    for d in disputes:
        t = Trade.query.get(d.trade_id)
        if not t:
            continue
        if d.ai_resolution_type == 'release_to_supplier' and t.supplier_id == user_id:
            disputes_won += 1
        elif d.ai_resolution_type == 'refund_to_buyer' and t.buyer_id == user_id:
            disputes_won += 1

    # ── Payment Reliability ──
    buyer_settled = [t for t in settled if t.buyer_id == user_id]
    if buyer_settled:
        # Base 60, +2 per clean buyer trade, capped 96
        payment_reliability = min(96.0, 60.0 + len(buyer_settled) * 2.0)
    elif total_trades > 0:
        payment_reliability = 68.0
    else:
        payment_reliability = 50.0

    # ── Delivery Accuracy ──
    supplier_settled  = [t for t in settled if t.supplier_id == user_id]
    supplier_disputes = [d for d in disputes
                         if Trade.query.get(d.trade_id) and
                         Trade.query.get(d.trade_id).supplier_id == user_id]
    if supplier_settled:
        clean_deliveries = len(supplier_settled) - len(supplier_disputes)
        delivery_accuracy = max(40.0, (clean_deliveries / len(supplier_settled)) * 100.0)
    elif total_trades > 0:
        delivery_accuracy = 70.0
    else:
        delivery_accuracy = 50.0

    # ── Dispute Rate (% of trades that became disputes) ──
    dispute_rate = (total_disputes / max(total_trades, 1)) * 100.0 if total_trades > 0 else 0.0

    # ── Dispute Win Rate ──
    dispute_win_rate = (disputes_won / max(total_disputes, 1)) * 100.0 if total_disputes > 0 else 100.0

    # ── Corridor Experience ──
    # Proxy: unique trade partners as indicator of multi-route experience
    partner_ids = set()
    for t in all_trades:
        partner_ids.add(t.buyer_id if t.supplier_id == user_id else t.supplier_id)
    corridor_experience = min(100.0, len(partner_ids) * 12.0)

    # ── Volume Score ──
    volume_score = min(100.0, all_trade_count * 4.0)

    # ── Trade Velocity (recency signal) ──
    # How many trades in the last 90 days
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=90)
    recent_trades = [t for t in all_trades if t.created_at and t.created_at >= cutoff]
    trade_velocity = min(100.0, len(recent_trades) * 10.0)

    # ── Clean Streak ──
    # Consecutive settled trades from most recent backward without a dispute
    dispute_trade_ids = {d.trade_id for d in disputes}
    clean_streak = 0
    for t in settled:  # already sorted desc
        if t.id in dispute_trade_ids:
            break
        clean_streak += 1

    return {
        'payment_reliability':  round(payment_reliability, 2),
        'delivery_accuracy':    round(delivery_accuracy, 2),
        'dispute_rate':         round(dispute_rate, 2),
        'dispute_win_rate':     round(dispute_win_rate, 2),
        'corridor_experience':  round(corridor_experience, 2),
        'volume_score':         round(volume_score, 2),
        'trade_velocity':       round(trade_velocity, 2),
        'clean_streak':         clean_streak,
        'total_trades':         total_trades,
        'total_disputes':       total_disputes,
        'disputes_won':         disputes_won,
        'all_trade_count':      all_trade_count,
    }


# ─────────────────────────────────────────────────────────────
# LAYER 2 — AI SCORING (Groq llama-3.3-70b-versatile)
#
# The AI receives the full behavioral signal profile and reasons
# about PATTERNS — not just values. It asks:
#   "What does this combination of signals predict about how
#    this business will behave in their NEXT trade?"
#
# This is the judgment a formula cannot make.
# ─────────────────────────────────────────────────────────────

def _build_scoring_prompt(signals: dict, user_context: dict) -> str:
    return f"""You are AfriFlow's Trust Score AI — a behavioral reliability predictor for African B2B trade.

Your job is NOT to summarize what happened. Your job is to analyze the PATTERN of these behavioral signals and predict how reliably this business will perform in their NEXT trade.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BUSINESS CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Business Type: {user_context.get('business_type', 'Unknown')}
Location: {user_context.get('location', 'Unknown')}
Platform Tenure: {user_context.get('tenure_days', 0)} days

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BEHAVIORAL SIGNAL PROFILE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Completed Trades:   {signals['total_trades']}
Active Trade Count:       {signals['all_trade_count']}
Recent (90-day) Trades:   {signals['trade_velocity'] / 10:.0f} trades

Payment Reliability:      {signals['payment_reliability']:.1f}/100
  (Does this buyer fund escrow promptly after agreement?)

Delivery Accuracy:        {signals['delivery_accuracy']:.1f}/100
  (Does this supplier deliver goods as described?)

Dispute Rate:             {signals['dispute_rate']:.1f}%
  ({signals['total_disputes']} disputes across {signals['total_trades']} settled trades)

Dispute Win Rate:         {signals['dispute_win_rate']:.1f}%
  (When disputed, evidence resolved in their favor {signals['disputes_won']} of {signals['total_disputes']} times)

Corridor Experience:      {signals['corridor_experience']:.1f}/100
  (Multi-partner trade experience indicator)

Trade Velocity:           {signals['trade_velocity']:.1f}/100
  (Recent activity — higher = actively trading)

Clean Streak:             {signals['clean_streak']} consecutive trades without dispute

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCORING INSTRUCTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Analyze these signals as a BEHAVIORAL PATTERN. Consider:

1. CONSISTENCY — Are signals consistent with each other, or do some contradict?
   A business with high delivery accuracy but rising dispute rate is showing early warning signs.

2. TRAJECTORY — What does the clean streak and velocity suggest about recent behavior?
   Recent improvement matters more than historical problems.

3. CONFIDENCE CALIBRATION — New businesses (few trades) should score conservatively.
   A business with 2 clean trades should not score 90+ regardless of accuracy metrics.

4. PATTERN RECOGNITION — Which combination of signals most predicts future reliability?
   High payment reliability + high delivery accuracy + low dispute rate = trustworthy.
   High volume + high dispute rate = growing but problematic.

5. DISPUTE QUALITY — A high win rate on disputes means disputes were legitimate, not abusive.
   Adjust your assessment accordingly.

Score range: 10–100
New business floor: max 60 for 0 trades, max 72 for 1-2 trades, max 85 for 3-5 trades.

You MUST respond with ONLY valid JSON:
{{
  "predicted_score": <integer 10-100>,
  "score_reasoning": "<2-3 sentences explaining the PRIMARY pattern driving this score>",
  "strongest_signal": "<which signal most drives this score and why>",
  "risk_flag": "<null OR one sentence describing the biggest reliability risk>",
  "trajectory": "<improving | stable | declining | insufficient_data>"
}}

No text outside the JSON."""


def score_from_signals_ai(signals: dict, user_context: dict) -> tuple:
    """
    Sends behavioral signals to Groq for pattern-based predictive scoring.
    Returns (score: float, ai_meta: dict) tuple.
    ai_meta contains reasoning, trajectory, risk flags.
    """
    client = Groq(api_key=GROQ_API_KEY)
    prompt = _build_scoring_prompt(signals, user_context)

    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {
                'role': 'system',
                'content': (
                    'You are AfriFlow Trust Score AI. '
                    'You respond only with valid JSON. '
                    'No markdown. No preamble. No explanation outside JSON structure.'
                )
            },
            {'role': 'user', 'content': prompt}
        ],
        temperature=0.05,   # Near-deterministic — scoring must be stable
        max_tokens=400
    )

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r'^```json\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    parsed = json.loads(raw)

    score = float(parsed.get('predicted_score', 50))
    score = max(SCORE_MIN, min(SCORE_MAX, score))

    return round(score, 1), {
        'reasoning':      parsed.get('score_reasoning', ''),
        'strongest':      parsed.get('strongest_signal', ''),
        'risk_flag':      parsed.get('risk_flag'),
        'trajectory':     parsed.get('trajectory', 'insufficient_data'),
        'source':         'groq_ai_scoring',
        'model':          TEXT_MODEL,
    }


# ─────────────────────────────────────────────────────────────
# LAYER 3 — DETERMINISTIC FALLBACK
# Used when: no API key, rate limit, or network error.
# Mirrors the signal interpretation logic of the AI prompt
# but expressed as weighted arithmetic.
# ─────────────────────────────────────────────────────────────

def score_from_signals_fallback(signals: dict) -> tuple:
    """Weighted deterministic scoring. Returns (score, meta) tuple."""

    s = signals

    # Weighted base
    base = (
        s['payment_reliability']  * 0.30 +
        s['delivery_accuracy']    * 0.30 +
        (100 - s['dispute_rate']) * 0.15 +
        s['dispute_win_rate']     * 0.10 +
        s['corridor_experience']  * 0.08 +
        s['trade_velocity']       * 0.07
    )

    # Clean streak bonus: +0.5 per consecutive clean trade, max +8
    streak_bonus = min(8.0, s['clean_streak'] * 0.5)
    raw = base + streak_bonus

    # Volume-based confidence floor
    if s['total_trades'] == 0:
        raw = min(raw, 55.0)
        trajectory = 'insufficient_data'
    elif s['total_trades'] <= 2:
        raw = min(raw, 70.0)
        trajectory = 'insufficient_data'
    elif s['total_trades'] <= 5:
        raw = min(raw, 82.0)
        trajectory = 'stable'
    else:
        trajectory = 'stable'

    score = round(max(SCORE_MIN, min(SCORE_MAX, raw)), 1)

    return score, {
        'reasoning':  'Score computed from weighted behavioral signals.',
        'strongest':  'payment_reliability and delivery_accuracy (combined 60% weight)',
        'risk_flag':  f"Dispute rate {s['dispute_rate']:.1f}% — monitor." if s['dispute_rate'] > 10 else None,
        'trajectory': trajectory,
        'source':     'deterministic_fallback',
        'model':      None,
    }


# ─────────────────────────────────────────────────────────────
# LAYER 4 — ORCHESTRATION + PERSISTENCE
# ─────────────────────────────────────────────────────────────

def calculate_and_update_trust_score(user_id: int) -> dict:
    """
    Full scoring pipeline. Called after every completed trade or dispute resolution.

    Flow:
      compute_signals() → [AI scoring | fallback] → persist → return dict
    """
    from models import User

    user = User.query.get(user_id)
    user_context = {}
    if user:
        tenure = (datetime.datetime.utcnow() - user.created_at).days if user.created_at else 0
        user_context = {
            'business_type': user.business_type,
            'location':      user.location,
            'tenure_days':   tenure,
        }

    signals = compute_signals(user_id)

    # Attempt AI scoring; fall back on any failure
    ai_meta = {}
    if GROQ_API_KEY:
        try:
            new_score, ai_meta = score_from_signals_ai(signals, user_context)
            print(f"[TrustScore] AI scored user {user_id}: {new_score} ({ai_meta.get('trajectory')})")
        except Exception as e:
            print(f"[TrustScore] AI scoring failed for user {user_id}: {e}. Using fallback.")
            new_score, ai_meta = score_from_signals_fallback(signals)
    else:
        print(f"[TrustScore] No API key. Using fallback for user {user_id}.")
        new_score, ai_meta = score_from_signals_fallback(signals)

    # Persist
    trust = TrustScore.query.filter_by(user_id=user_id).first()
    if not trust:
        trust = TrustScore(user_id=user_id)
        db.session.add(trust)

    trust.previous_score          = trust.overall_score
    trust.overall_score           = new_score
    trust.payment_reliability     = signals['payment_reliability']
    trust.delivery_accuracy       = signals['delivery_accuracy']
    trust.dispute_rate_value      = signals['dispute_rate']
    trust.corridor_experience_value = signals['corridor_experience']
    trust.total_trades            = signals['total_trades']
    trust.total_disputes          = signals['total_disputes']
    trust.disputes_won            = signals['disputes_won']
    trust.score_reasoning         = ai_meta.get('reasoning', '')
    trust.score_trajectory        = ai_meta.get('trajectory', 'insufficient_data')
    trust.score_risk_flag         = ai_meta.get('risk_flag')
    trust.score_source            = ai_meta.get('source', 'unknown')
    trust.updated_at              = datetime.datetime.utcnow()

    db.session.commit()
    return trust.to_dict()


def get_trust_profile(user_id: int) -> dict:
    trust = TrustScore.query.filter_by(user_id=user_id).first()
    if not trust:
        return calculate_and_update_trust_score(user_id)
    return trust.to_dict()