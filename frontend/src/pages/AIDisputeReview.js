// FILE: frontend/src/pages/AIDisputeReview.js

import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import api from '../api/client';

// ─────────────────────────────────────────────────────────────
// Two-phase analysis steps
// Phase 1: Vision model reads images
// Phase 2: Text model adjudicates all evidence
// ─────────────────────────────────────────────────────────────

const PHASE_1_STEPS = [
  { key: 'load_images',    label: 'Loading image evidence files' },
  { key: 'vision_init',    label: 'Initialising llama-3.2-11b-vision-preview' },
  { key: 'vision_read',    label: 'Vision model reading image evidence' },
  { key: 'vision_extract', label: 'Extracting visual observations' },
  { key: 'vision_done',    label: 'Visual findings compiled' },
];

const PHASE_2_STEPS = [
  { key: 'load_text',      label: 'Reading text evidence from both parties' },
  { key: 'read_terms',     label: 'Parsing trade agreement terms' },
  { key: 'read_profiles',  label: 'Evaluating trust profiles and trajectories' },
  { key: 'synthesise',     label: 'Synthesising text + visual evidence' },
  { key: 'adjudicate',     label: 'Computing confidence score and resolution' },
];

const VISUAL_IMPACT_CONFIG = {
  none:          { label: 'No image evidence',      color: 'text-gray-400',   bg: 'bg-gray-800' },
  supporting:    { label: 'Images support finding', color: 'text-blue-400',   bg: 'bg-blue-900/30' },
  decisive:      { label: 'Images were decisive',   color: 'text-green-400',  bg: 'bg-green-900/30' },
  contradicting: { label: 'Images contradicted text', color: 'text-red-400',  bg: 'bg-red-900/30' },
};

export default function AIDisputeReview() {
  const { disputeId } = useParams();
  const navigate = useNavigate();

  const [dispute, setDispute]   = useState(null);
  const [evidence, setEvidence] = useState([]);
  const [tradeId, setTradeId]   = useState(null);

  const [phase, setPhase]             = useState(0);   // 0=idle 1=phase1 2=phase2 3=done
  const [phase1Step, setPhase1Step]   = useState(-1);
  const [phase2Step, setPhase2Step]   = useState(-1);
  const [result, setResult]           = useState(null);
  const [expandedFindings, setExpandedFindings] = useState(false);

  useEffect(() => {
    api.get(`/disputes/${disputeId}`).then(res => {
      setDispute(res.data.dispute);
      setEvidence(res.data.evidence);
      setTradeId(res.data.dispute.trade_id);
      // If already reviewed, show result immediately
      if (res.data.dispute.ai_confidence) {
        setResult({
          confidence:            res.data.dispute.ai_confidence,
          finding:               res.data.dispute.ai_finding,
          recommendation:        res.data.dispute.ai_recommendation,
          resolution_type:       res.data.dispute.ai_resolution_type,
          visual_findings:       res.data.dispute.ai_visual_findings || [],
          visual_evidence_impact: res.data.dispute.ai_visual_impact || 'none',
          vision_model:          res.data.dispute.ai_vision_model,
          text_model:            res.data.dispute.ai_text_model,
        });
        setPhase(3);
      }
    });
  }, [disputeId]);

  const hasImages = evidence.some(e => e.file_path);

  const runReview = async () => {
    // ── Animate Phase 1 ──
    setPhase(1);
    for (let i = 0; i < PHASE_1_STEPS.length; i++) {
      setPhase1Step(i);
      await delay(hasImages ? 700 + Math.random() * 500 : 300);
    }

    // ── Animate Phase 2 ──
    setPhase(2);
    for (let i = 0; i < PHASE_2_STEPS.length; i++) {
      setPhase2Step(i);
      await delay(600 + Math.random() * 400);
    }

    // ── API call (happens during animation) ──
    try {
      const res = await api.post(`/disputes/${disputeId}/review`);
      setResult(res.data.ai_result);
      setDispute(res.data.dispute);
      setPhase(3);
    } catch (e) {
      alert('AI review failed: ' + (e.response?.data?.error || e.message));
      setPhase(0);
    }
  };

  if (!dispute) return null;

  const isAutoResolved = result && result.confidence >= 85;

  const resolutionStyle = {
    release_to_supplier: { color: 'text-green-400', icon: '💰', label: 'Release to Supplier' },
    refund_to_buyer:     { color: 'text-blue-400',  icon: '↩️', label: 'Refund to Buyer'     },
    partial_refund:      { color: 'text-yellow-400',icon: '⚖️', label: 'Partial Refund'       },
    escalate_to_human:   { color: 'text-orange-400',icon: '👤', label: 'Escalate to Human'    },
  };

  const resStyle = result ? (resolutionStyle[result.resolution_type] || resolutionStyle['escalate_to_human']) : null;
  const visImpact = result ? (VISUAL_IMPACT_CONFIG[result.visual_evidence_impact] || VISUAL_IMPACT_CONFIG['none']) : null;

  return (
    <div className="max-w-2xl mx-auto px-6 py-8 fade-in space-y-5">

      {/* Header */}
      <div>
        <span className="text-xs font-medium text-red-400 bg-red-900/30 px-2 py-0.5 rounded-full">
          Dispute #{dispute.id}
        </span>
        <h1 className="text-2xl font-bold text-white mt-2">AI Dispute Review</h1>
        <p className="text-gray-500 text-sm mt-0.5">
          Two-phase analysis: vision model reads images → text model adjudicates all evidence.
        </p>
      </div>

      {/* Pipeline diagram */}
      <div className="card space-y-3">
        <div className="flex items-stretch gap-0">
          <PipelinePhase
            number={1}
            title="Vision Analysis"
            model="llama-3.2-11b-vision-preview"
            description="Reads submitted photos and extracts objective visual findings"
            active={phase === 1}
            done={phase >= 2}
            hasContent={hasImages}
            noContentLabel="No images submitted"
          />
          <div className="flex items-center px-2">
            <div className={`flex-shrink-0 transition-colors ${phase >= 2 ? 'text-brand-400' : 'text-gray-700'}`}>→</div>
          </div>
          <PipelinePhase
            number={2}
            title="Final Adjudication"
            model="llama-3.3-70b-versatile"
            description="Synthesises all evidence, trust profiles, and visual findings"
            active={phase === 2}
            done={phase >= 3}
            hasContent={true}
            noContentLabel=""
          />
        </div>
      </div>

      {/* Dispute summary */}
      <div className="card space-y-2">
        <p className="text-xs text-gray-500 uppercase tracking-wide">Dispute</p>
        <p className="font-medium text-white">{dispute.reason}</p>
        {dispute.description && <p className="text-sm text-gray-400">{dispute.description}</p>}
      </div>

      {/* Evidence preview */}
      {evidence.length > 0 && (
        <div className="card space-y-3">
          <p className="text-xs text-gray-500 uppercase tracking-wide">
            Evidence Submitted ({evidence.length} items — {evidence.filter(e => e.file_path).length} images)
          </p>
          {evidence.map((e, i) => (
            <div key={i} className="bg-surface rounded-xl p-3 space-y-1">
              <div className="flex items-center gap-2 flex-wrap">
                <span className={`text-xs px-2 py-0.5 rounded-full ${
                  e.submitted_by === dispute.raised_by
                    ? 'bg-blue-900/30 text-blue-400'
                    : 'bg-green-900/30 text-green-400'
                }`}>
                  {e.submitted_by === dispute.raised_by ? 'Disputing Party' : 'Responding Party'}
                </span>
                <span className="text-xs text-gray-600 capitalize">{e.evidence_type?.replace(/_/g, ' ')}</span>
                {e.file_path && (
                  <span className="text-xs bg-purple-900/30 text-purple-400 px-2 py-0.5 rounded-full">
                    📷 Image
                  </span>
                )}
              </div>
              <p className="text-sm text-gray-300">{e.content || '(Image evidence)'}</p>
            </div>
          ))}
        </div>
      )}

      {/* ── Analysis UI ── */}
      {phase === 0 && !result && (
        <div className="card text-center space-y-4 py-8">
          <div className="text-5xl">🤖</div>
          <div>
            <h3 className="font-semibold text-white">Ready to Analyse</h3>
            <p className="text-gray-500 text-sm mt-1">
              {hasImages
                ? 'The vision model will analyse submitted images, then the text model will adjudicate.'
                : 'No images detected. Text model will adjudicate directly from evidence and trust profiles.'}
            </p>
          </div>
          <button onClick={runReview} className="btn-primary px-8 mx-auto">
            Run AI Analysis →
          </button>
        </div>
      )}

      {/* Phase 1 in progress */}
      {phase === 1 && (
        <AnalysisCard
          title="Phase 1 — Vision Analysis"
          subtitle="llama-3.2-11b-vision-preview reading image evidence"
          steps={PHASE_1_STEPS}
          currentStep={phase1Step}
          color="purple"
        />
      )}

      {/* Phase 2 in progress */}
      {phase === 2 && (
        <div className="space-y-3">
          <AnalysisCard
            title="Phase 1 — Vision Analysis"
            subtitle="llama-3.2-11b-vision-preview"
            steps={PHASE_1_STEPS}
            currentStep={PHASE_1_STEPS.length}
            color="purple"
            done
          />
          <AnalysisCard
            title="Phase 2 — Final Adjudication"
            subtitle="llama-3.3-70b-versatile synthesising all evidence"
            steps={PHASE_2_STEPS}
            currentStep={phase2Step}
            color="brand"
          />
        </div>
      )}

      {/* ── Result ── */}
      {result && phase === 3 && (
        <div className="space-y-4 fade-in">

          {/* Models used */}
          <div className="flex gap-2 flex-wrap">
            {result.vision_model && (
              <span className="text-xs bg-purple-900/30 text-purple-400 border border-purple-800/30 px-3 py-1 rounded-full">
                📷 {result.vision_model}
              </span>
            )}
            {result.text_model && (
              <span className="text-xs bg-blue-900/30 text-blue-400 border border-blue-800/30 px-3 py-1 rounded-full">
                🧠 {result.text_model}
              </span>
            )}
            {result.source && (
              <span className="text-xs bg-gray-800 text-gray-500 px-3 py-1 rounded-full">
                {result.source === 'groq_two_phase' ? '✓ Two-phase AI pipeline' : '⚡ Fallback engine'}
              </span>
            )}
          </div>

          {/* Main result card */}
          <div className={`card space-y-5 ${isAutoResolved ? 'border-green-800/40' : 'border-yellow-800/40'}`}>

            {/* Confidence + resolution status */}
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-gray-500 uppercase tracking-wide mb-1">AI Confidence</p>
                <div className="flex items-center gap-3">
                  <span className={`text-4xl font-bold ${isAutoResolved ? 'text-green-400' : 'text-yellow-400'}`}>
                    {result.confidence}%
                  </span>
                  <span className={`status-badge text-xs ${
                    isAutoResolved
                      ? 'bg-green-900/40 text-green-400'
                      : 'bg-yellow-900/40 text-yellow-400'
                  }`}>
                    {isAutoResolved ? 'AUTO-RESOLVED ✓' : 'ESCALATED TO HUMAN'}
                  </span>
                </div>
              </div>
              <ConfidenceRing value={result.confidence} />
            </div>

            {/* Confidence bar with 85% threshold marker */}
            <div>
              <div className="flex justify-between text-xs text-gray-600 mb-1">
                <span>0%</span>
                <span className="text-yellow-600">85% auto-resolve threshold</span>
                <span>100%</span>
              </div>
              <div className="h-3 bg-surface rounded-full relative overflow-hidden">
                <div
                  className={`h-full rounded-full progress-fill ${result.confidence >= 85 ? 'bg-green-500' : 'bg-yellow-500'}`}
                  style={{ width: `${result.confidence}%` }}
                />
                <div className="absolute top-0 bottom-0 w-px bg-yellow-600/50" style={{ left: '85%' }} />
              </div>
            </div>

            {/* Visual evidence impact badge */}
            {visImpact && (
              <div className={`flex items-center gap-2 px-3 py-2 rounded-xl ${visImpact.bg}`}>
                <span>📷</span>
                <span className={`text-sm font-medium ${visImpact.color}`}>{visImpact.label}</span>
              </div>
            )}

            {/* Finding */}
            <div className="bg-surface rounded-xl p-4 space-y-2">
              <p className="text-xs text-gray-500 uppercase tracking-wide">AI Finding</p>
              <p className="text-sm text-gray-200 leading-relaxed">{result.finding}</p>
            </div>

            {/* Recommendation */}
            <div className="bg-surface rounded-xl p-4 space-y-2">
              <p className="text-xs text-gray-500 uppercase tracking-wide">Recommendation</p>
              <p className={`font-semibold ${resStyle?.color}`}>{result.recommendation}</p>
            </div>

            {/* Resolution type */}
            <div className="flex items-center gap-3 p-4 bg-surface rounded-xl">
              <span className="text-2xl">{resStyle?.icon}</span>
              <div>
                <p className="text-xs text-gray-500">Resolution</p>
                <p className={`font-semibold ${resStyle?.color}`}>{resStyle?.label}</p>
              </div>
            </div>

            {/* Visual findings detail */}
            {result.visual_findings?.length > 0 && (
              <div className="space-y-2">
                <button
                  onClick={() => setExpandedFindings(!expandedFindings)}
                  className="text-xs text-purple-400 hover:text-purple-300 transition-colors flex items-center gap-1"
                >
                  📷 {expandedFindings ? 'Hide' : 'Show'} vision model findings ({result.visual_findings.length} image{result.visual_findings.length > 1 ? 's' : ''})
                </button>
                {expandedFindings && (
                  <div className="space-y-3 fade-in">
                    {result.visual_findings.map((vf, i) => (
                      <VisionFindingCard key={i} finding={vf} index={i} />
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Reasoning steps */}
            {result.reasoning_steps?.length > 0 && (
              <details className="group">
                <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-400 transition-colors">
                  View adjudication reasoning ({result.reasoning_steps.length} steps)
                </summary>
                <div className="mt-3 space-y-2">
                  {result.reasoning_steps.map((step, i) => (
                    <div key={i} className="flex items-start gap-2 text-xs text-gray-500">
                      <span className="text-brand-600 mt-0.5 flex-shrink-0">{i + 1}.</span>
                      <span>{step}</span>
                    </div>
                  ))}
                </div>
              </details>
            )}

            <button onClick={() => navigate(`/trade/${tradeId}`)} className="btn-primary w-full text-center">
              Back to Trade →
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Sub-components ────────────────────────────────────────────

function PipelinePhase({ number, title, model, description, active, done, hasContent, noContentLabel }) {
  return (
    <div className={`flex-1 p-4 rounded-xl border transition-all
      ${active ? 'border-brand-500 bg-brand-900/10' :
        done   ? 'border-green-800/40 bg-green-900/5' :
                 'border-border bg-surface'}`}>
      <div className="flex items-center gap-2 mb-2">
        <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0
          ${done ? 'bg-green-500 text-white' : active ? 'bg-brand-500 text-white pulse-soft' : 'bg-border text-gray-600'}`}>
          {done ? '✓' : number}
        </div>
        <p className={`text-sm font-semibold ${done ? 'text-green-400' : active ? 'text-white' : 'text-gray-500'}`}>
          {title}
        </p>
      </div>
      <p className="text-xs text-gray-600 font-mono mb-1">{model}</p>
      <p className="text-xs text-gray-500">{description}</p>
      {!hasContent && (
        <p className="text-xs text-gray-700 mt-1 italic">{noContentLabel}</p>
      )}
    </div>
  );
}

function AnalysisCard({ title, subtitle, steps, currentStep, color, done }) {
  const activeColor = color === 'purple' ? 'bg-purple-500' : 'bg-brand-500';
  const doneColor   = color === 'purple' ? 'text-purple-400' : 'text-brand-400';

  return (
    <div className="card space-y-4">
      <div className="flex items-center gap-3">
        {!done && (
          <div className="flex gap-1">
            <div className={`w-2 h-2 rounded-full dot-bounce ${activeColor}`} />
            <div className={`w-2 h-2 rounded-full dot-bounce ${activeColor}`} />
            <div className={`w-2 h-2 rounded-full dot-bounce ${activeColor}`} />
          </div>
        )}
        {done && <span className="text-green-400 text-lg">✓</span>}
        <div>
          <p className={`font-medium text-sm ${done ? 'text-green-400' : 'text-white'}`}>{title}</p>
          <p className="text-xs text-gray-500">{subtitle}</p>
        </div>
      </div>
      <div className="space-y-2">
        {steps.map((step, i) => {
          const isComplete = i < currentStep || done;
          const isCurrent  = i === currentStep && !done;
          return (
            <div key={step.key} className="flex items-center gap-3">
              <div className={`w-5 h-5 rounded-full flex items-center justify-center text-xs flex-shrink-0
                ${isComplete ? 'bg-green-500 text-white' :
                  isCurrent  ? `${activeColor} text-white pulse-soft` :
                               'bg-border text-gray-600'}`}>
                {isComplete ? '✓' : isCurrent ? '·' : ''}
              </div>
              <span className={`text-sm transition-colors
                ${isComplete ? 'text-green-400' :
                  isCurrent  ? 'text-white' : 'text-gray-600'}`}>
                {step.label}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function VisionFindingCard({ finding, index }) {
  const packageColor = {
    intact:      'text-green-400',
    damaged:     'text-red-400',
    opened:      'text-yellow-400',
    not_visible: 'text-gray-500',
  };

  return (
    <div className="bg-surface border border-purple-800/20 rounded-xl p-4 space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-xs bg-purple-900/40 text-purple-400 px-2 py-0.5 rounded-full">
            📷 Image {index + 1}
          </span>
          <span className={`text-xs px-2 py-0.5 rounded-full ${
            finding.submitter_role === 'buyer'
              ? 'bg-blue-900/30 text-blue-400'
              : 'bg-green-900/30 text-green-400'
          }`}>
            {finding.submitter_role}
          </span>
        </div>
        <span className={`text-xs ${finding.image_quality === 'clear' ? 'text-green-400' : 'text-yellow-400'}`}>
          {finding.image_quality} quality
        </span>
      </div>

      <p className="text-sm text-gray-300">{finding.visual_finding}</p>

      {finding.key_observations?.length > 0 && (
        <ul className="space-y-1">
          {finding.key_observations.map((obs, i) => (
            <li key={i} className="text-xs text-gray-500 flex items-start gap-1.5">
              <span className="text-purple-600 mt-0.5">•</span>
              <span>{obs}</span>
            </li>
          ))}
        </ul>
      )}

      <div className="flex flex-wrap gap-3 text-xs">
        {finding.damage_visible !== null && finding.damage_visible !== undefined && (
          <span className={finding.damage_visible ? 'text-red-400' : 'text-green-400'}>
            {finding.damage_visible ? '⚠ Damage visible' : '✓ No damage visible'}
          </span>
        )}
        {finding.goods_match_description !== null && finding.goods_match_description !== undefined && (
          <span className={finding.goods_match_description ? 'text-green-400' : 'text-yellow-400'}>
            {finding.goods_match_description ? '✓ Goods match description' : '⚠ Goods may not match'}
          </span>
        )}
        {finding.packaging_condition && finding.packaging_condition !== 'not_visible' && (
          <span className={packageColor[finding.packaging_condition] || 'text-gray-400'}>
            📦 Packaging: {finding.packaging_condition}
          </span>
        )}
        {finding.supports_claimant !== null && finding.supports_claimant !== undefined && (
          <span className={finding.supports_claimant ? 'text-green-400' : 'text-red-400'}>
            {finding.supports_claimant ? '✓ Supports submitter' : '✗ Does not support submitter'}
          </span>
        )}
      </div>
    </div>
  );
}

function ConfidenceRing({ value }) {
  const r = 32;
  const circ = 2 * Math.PI * r;
  const offset = circ - (value / 100) * circ;
  const color = value >= 85 ? '#22c55e' : value >= 70 ? '#f59e0b' : '#ef4444';
  return (
    <svg width="80" height="80" viewBox="0 0 80 80" className="-rotate-90 flex-shrink-0">
      <circle cx="40" cy="40" r={r} fill="none" stroke="#2a2d3e" strokeWidth="7" />
      <circle cx="40" cy="40" r={r} fill="none" stroke={color} strokeWidth="7"
        strokeDasharray={circ} strokeDashoffset={offset} strokeLinecap="round"
        style={{ transition: 'stroke-dashoffset 1.2s ease-out' }}
      />
    </svg>
  );
}

function delay(ms) { return new Promise(r => setTimeout(r, ms)); }