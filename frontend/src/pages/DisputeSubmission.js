// FILE: frontend/src/pages/DisputeSubmission.js

import React, { useEffect, useState, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import api from '../api/client';

const DISPUTE_REASONS = [
  'Goods arrived damaged',
  'Goods not delivered',
  'Wrong goods received',
  'Goods do not match description',
  'Partial delivery only',
  'Quality significantly below agreed standard',
  'Other',
];

// ─────────────────────────────────────────────────────────────
// Evidence Form — handles both text and image evidence.
// Images are uploaded via multipart form. Text via JSON.
// The vision model (llama-3.2-11b-vision-preview) processes images.
// ─────────────────────────────────────────────────────────────

function EvidenceForm({ disputeId, tradeDescription, disputeReason, onSubmitted }) {
  const [items, setItems]   = useState([{ type: 'text', content: '', file: null, preview: null, analysis: null, analysing: false }]);
  const [loading, setLoading] = useState(false);
  const fileRefs = useRef([]);

  const addItem = () => setItems(prev => [
    ...prev, { type: 'text', content: '', file: null, preview: null, analysis: null, analysing: false }
  ]);

  const removeItem = (i) => setItems(prev => prev.filter((_, idx) => idx !== i));

  const updateItem = (i, updates) => {
    setItems(prev => prev.map((item, idx) => idx === i ? { ...item, ...updates } : item));
  };

  const handleFileChange = async (i, e) => {
    const file = e.target.files[0];
    if (!file) return;

    const preview = URL.createObjectURL(file);
    updateItem(i, { file, preview, type: 'image', content: `Photo evidence: ${file.name}`, analysis: null, analysing: true });

    // ── Run AI image analysis immediately ─────
    try {
      const form = new FormData();
      form.append('file',    file);
      form.append('context', `A trade dispute about ${tradeDescription || 'goods'}`);
      form.append('purpose', disputeReason || 'dispute evidence');

      const res = await api.post('/verify/image', form, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      updateItem(i, { analysing: false, analysis: res.data.analysis });
    } catch {
      updateItem(i, { analysing: false, analysis: { error: true, description: 'Image analysis unavailable — will still be submitted.' } });
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      for (const item of items) {
        if (item.file) {
          // Image evidence — multipart upload
          const formData = new FormData();
          formData.append('file', item.file);
          formData.append('evidence_type', 'image');
          formData.append('content', item.content || `Photo: ${item.file.name}`);
          await api.post(`/disputes/${disputeId}/evidence`, formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
          });
        } else if (item.content.trim()) {
          // Text evidence — JSON
          await api.post(`/disputes/${disputeId}/evidence`, {
            evidence_type: item.type,
            content: item.content,
          });
        }
      }
      onSubmitted();
    } catch (err) {
      alert(err.response?.data?.error || 'Failed to submit evidence.');
    } finally {
      setLoading(false);
    }
  };

  const hasContent = items.some(item => item.file || item.content.trim());

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {/* Vision model note */}
      <div className="bg-purple-900/10 border border-purple-800/30 rounded-xl p-3 flex items-start gap-3">
        <span className="text-purple-400 text-lg">📷</span>
        <div>
          <p className="text-sm text-purple-300 font-medium">Images analyzed by llama-3.2-11b-vision-preview</p>
          <p className="text-xs text-purple-500 mt-0.5">
            Upload photos of goods, packaging, or delivery condition. The vision model will extract objective observations from your images.
          </p>
        </div>
      </div>

      {items.map((item, i) => (
        <div key={i} className="card space-y-3 relative">
          {items.length > 1 && (
            <button
              type="button"
              onClick={() => removeItem(i)}
              className="absolute top-3 right-3 text-gray-600 hover:text-red-400 transition-colors text-sm"
            >
              ✕
            </button>
          )}

          {/* Type selector */}
          <div className="flex gap-2 flex-wrap">
            {[
              { value: 'text',       label: '📝 Written' },
              { value: 'tracking',   label: '📦 Tracking' },
              { value: 'image',      label: '📷 Photo'   },
              { value: 'communication', label: '💬 Messages' },
            ].map(t => (
              <button
                key={t.value}
                type="button"
                onClick={() => {
                  updateItem(i, { type: t.value, file: null, preview: null });
                  if (t.value !== 'image') updateItem(i, { file: null, preview: null });
                }}
                className={`text-xs px-3 py-1.5 rounded-full border transition-all
                  ${item.type === t.value
                    ? 'border-brand-500 bg-brand-900/20 text-brand-400'
                    : 'border-border text-gray-500 hover:border-gray-600'}`}
              >
                {t.label}
              </button>
            ))}
          </div>

          {/* Image upload */}
          {item.type === 'image' ? (
            <div>
              <input
                type="file"
                accept="image/jpeg,image/png,image/gif,image/webp"
                ref={el => fileRefs.current[i] = el}
                onChange={e => handleFileChange(i, e)}
                className="hidden"
              />
              {item.preview ? (
                <div className="space-y-3">
                  <div className="relative">
                    <img
                      src={item.preview}
                      alt="Evidence preview"
                      className="w-full max-h-48 object-cover rounded-xl border border-border"
                    />
                    <button
                      type="button"
                      onClick={() => {
                        updateItem(i, { file: null, preview: null, content: '', analysis: null, analysing: false });
                        if (fileRefs.current[i]) fileRefs.current[i].value = '';
                      }}
                      className="absolute top-2 right-2 bg-black/60 text-white text-xs px-2 py-1 rounded-lg hover:bg-red-900/80 transition-colors"
                    >
                      Remove
                    </button>
                  </div>
                  <input
                    className="input text-sm"
                    placeholder="Describe what this image shows..."
                    value={item.content}
                    onChange={e => updateItem(i, { content: e.target.value })}
                  />
                  {/* Analysing spinner */}
                  {item.analysing && (
                    <div className="flex items-center gap-2 bg-purple-900/20 border border-purple-800/30 rounded-xl p-3">
                      <div className="flex gap-1">
                        <div className="w-1.5 h-1.5 bg-purple-400 rounded-full dot-bounce" />
                        <div className="w-1.5 h-1.5 bg-purple-400 rounded-full dot-bounce" />
                        <div className="w-1.5 h-1.5 bg-purple-400 rounded-full dot-bounce" />
                      </div>
                      <span className="text-xs text-purple-300">Vision model reading image...</span>
                      <span className="text-xs text-purple-600 ml-auto">llama-3.2-11b-vision-preview</span>
                    </div>
                  )}
                  {/* Analysis result */}
                  {item.analysis && !item.analysing && !item.analysis.error && (
                    <ImageAnalysisPreview analysis={item.analysis} />
                  )}
                  {item.analysis?.error && (
                    <p className="text-xs text-gray-500 italic">{item.analysis.description}</p>
                  )}
                </div>
              ) : (
                <button
                  type="button"
                  onClick={() => fileRefs.current[i]?.click()}
                  className="w-full border-2 border-dashed border-border hover:border-purple-600 rounded-xl p-8 text-center transition-colors group"
                >
                  <div className="text-4xl mb-2 group-hover:scale-110 transition-transform">📷</div>
                  <p className="text-sm text-gray-400">Click to upload photo evidence</p>
                  <p className="text-xs text-gray-600 mt-1">Analyzed instantly by llama-3.2-11b-vision-preview</p>
                </button>
              )}
            </div>
          ) : (
            <textarea
              className="input resize-none"
              rows={4}
              placeholder={
                item.type === 'tracking'
                  ? 'Paste tracking number, courier name, and delivery status...'
                  : item.type === 'communication'
                  ? 'Paste relevant messages, emails, or conversation records...'
                  : 'Describe your evidence in detail. Include dates, quantities, specific items...'
              }
              value={item.content}
              onChange={e => updateItem(i, { content: e.target.value })}
            />
          )}
        </div>
      ))}

      <button
        type="button"
        onClick={addItem}
        className="w-full border border-dashed border-border hover:border-brand-500 rounded-xl py-3 text-sm text-gray-500 hover:text-brand-400 transition-all"
      >
        + Add another evidence item
      </button>

      <button
        type="submit"
        disabled={loading || !hasContent}
        className="btn-primary w-full text-center disabled:opacity-50"
      >
        {loading ? (
          <span className="flex items-center justify-center gap-2">
            <div className="w-2 h-2 bg-white rounded-full dot-bounce" />
            <div className="w-2 h-2 bg-white rounded-full dot-bounce" />
            <div className="w-2 h-2 bg-white rounded-full dot-bounce" />
          </span>
        ) : 'Submit Evidence →'}
      </button>
    </form>
  );
}

export default function DisputeSubmission() {
  const { tradeId } = useParams();
  const navigate = useNavigate();
  const [trade, setTrade] = useState(null);
  const [step, setStep] = useState(1);
  const [dispute, setDispute] = useState(null);
  const [form, setForm] = useState({ reason: '', description: '' });
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    api.get(`/trades/${tradeId}`).then(res => setTrade(res.data.trade));
  }, [tradeId]);

  const handleRaiseDispute = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await api.post(`/trades/${tradeId}/dispute`, form);
      setDispute(res.data.dispute);
      setStep(2);
    } catch (e) {
      alert(e.response?.data?.error || 'Failed to raise dispute.');
    } finally {
      setLoading(false);
    }
  };

  if (!trade) return null;

  return (
    <div className="max-w-lg mx-auto px-6 py-8 fade-in space-y-5">
      {/* Progress */}
      <div className="flex items-center gap-2 mb-6">
        {[1, 2, 3].map(s => (
          <React.Fragment key={s}>
            <div className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold
              ${s < step ? 'bg-green-500 text-white' : s === step ? 'bg-brand-500 text-white' : 'bg-border text-gray-600'}`}>
              {s < step ? '✓' : s}
            </div>
            {s < 3 && <div className={`flex-1 h-px ${s < step ? 'bg-green-500' : 'bg-border'}`} />}
          </React.Fragment>
        ))}
      </div>

      {step === 1 && (
        <>
          <div>
            <h1 className="text-2xl font-bold text-white">Raise a Dispute</h1>
            <p className="text-gray-500 text-sm mt-1">Describe what went wrong with this trade.</p>
          </div>
          <form onSubmit={handleRaiseDispute} className="card space-y-4">
            <div>
              <label className="label">Reason</label>
              <select
                className="input"
                value={form.reason}
                onChange={e => setForm(f => ({ ...f, reason: e.target.value }))}
                required
              >
                <option value="">Select a reason...</option>
                {DISPUTE_REASONS.map(r => (
                  <option key={r} value={r}>{r}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="label">Description</label>
              <textarea
                className="input resize-none"
                rows={4}
                placeholder="Describe the issue in detail. The more specific you are, the better the AI can assess your case."
                value={form.description}
                onChange={e => setForm(f => ({ ...f, description: e.target.value }))}
                required
              />
            </div>
            <button type="submit" disabled={loading || !form.reason} className="btn-primary w-full text-center">
              {loading ? 'Submitting...' : 'Raise Dispute →'}
            </button>
          </form>
        </>
      )}

      {step === 2 && dispute && (
        <>
          <div>
            <div className="flex items-center gap-2 mb-2">
              <span className="text-orange-400 text-sm font-medium bg-orange-900/30 px-2 py-0.5 rounded-full">Dispute #{dispute.id}</span>
            </div>
            <h1 className="text-2xl font-bold text-white">Submit Your Evidence</h1>
            <p className="text-gray-500 text-sm mt-1">
              Images go to the vision model. Text goes to the adjudication model. Submit both for the strongest case.
            </p>
          </div>
          <EvidenceForm
            disputeId={dispute.id}
            tradeDescription={trade?.description}
            disputeReason={form.reason}
            onSubmitted={() => setStep(3)}
          />
        </>
      )}

      {step === 3 && (
        <div className="card text-center space-y-5">
          <div className="text-5xl">⚖️</div>
          <h2 className="text-2xl font-bold text-white">Evidence Submitted</h2>
          <p className="text-gray-400 text-sm">
            The other party can now submit their evidence. Once both sides have submitted, the AI will analyze the case.
          </p>
          <div className="bg-surface rounded-xl p-4 text-sm text-gray-400 text-left">
            <p className="font-medium text-white mb-2">What happens next:</p>
            <ul className="space-y-1 text-xs">
              <li>• Other party submits their evidence</li>
              <li>• AI reads and analyzes all evidence</li>
              <li>• Confidence ≥85% → auto-resolved in minutes</li>
              <li>• Confidence &lt;85% → human arbitrator reviews</li>
            </ul>
          </div>
          <div className="flex gap-3">
            <button onClick={() => navigate(`/dispute/${dispute.id}/review`)} className="btn-primary flex-1 text-center">
              Run AI Review Now →
            </button>
            <button onClick={() => navigate(`/trade/${tradeId}`)} className="btn-secondary flex-1 text-center">
              Back to Trade
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// ── ImageAnalysisPreview ──────────────────────────────────────
// Renders the instant AI image analysis result inline.
// Called after every image upload in EvidenceForm.

function ImageAnalysisPreview({ analysis }) {
  if (!analysis) return null;
  const authConf = analysis.authenticity_confidence || 0;
  const relConf  = analysis.relevance_confidence    || 0;

  return (
    <div className="bg-purple-900/10 border border-purple-800/30 rounded-xl p-4 space-y-3 fade-in">
      <div className="flex items-center gap-2">
        <span className="text-xs font-medium text-purple-300">🤖 AI Image Analysis</span>
        <span className="text-xs text-purple-600 ml-auto">llama-3.2-11b-vision-preview</span>
      </div>

      {/* Description */}
      <p className="text-sm text-gray-300 leading-relaxed">{analysis.description}</p>

      {/* Key details */}
      {analysis.key_details?.length > 0 && (
        <ul className="space-y-1">
          {analysis.key_details.map((d, i) => (
            <li key={i} className="text-xs text-gray-500 flex items-start gap-1.5">
              <span className="text-purple-600 mt-0.5 flex-shrink-0">•</span>
              <span>{d}</span>
            </li>
          ))}
        </ul>
      )}

      {/* Confidence bars */}
      <div className="grid grid-cols-2 gap-3">
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-gray-500">Authenticity</span>
            <span className={authConf >= 70 ? 'text-green-400' : 'text-yellow-400'}>{authConf}%</span>
          </div>
          <div className="h-1.5 bg-border rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full progress-fill ${authConf >= 70 ? 'bg-green-500' : 'bg-yellow-500'}`}
              style={{ width: `${authConf}%` }}
            />
          </div>
        </div>
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-gray-500">Relevance</span>
            <span className={relConf >= 70 ? 'text-green-400' : 'text-yellow-400'}>{relConf}%</span>
          </div>
          <div className="h-1.5 bg-border rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full progress-fill ${relConf >= 70 ? 'bg-green-500' : 'bg-yellow-500'}`}
              style={{ width: `${relConf}%` }}
            />
          </div>
        </div>
      </div>

      {/* Visible text extracted */}
      {analysis.visible_text && (
        <div className="bg-surface rounded-lg p-2 flex items-start gap-2">
          <span className="text-xs text-gray-500 flex-shrink-0">Text in image:</span>
          <span className="text-xs text-white font-mono">{analysis.visible_text}</span>
        </div>
      )}

      {/* Real photo badge */}
      <div className="flex flex-wrap gap-2 text-xs">
        {analysis.is_real_photograph === true && (
          <span className="text-green-400">✓ Real photograph</span>
        )}
        {analysis.is_real_photograph === false && (
          <span className="text-yellow-400">⚠ May not be a real photo</span>
        )}
        {analysis.is_relevant_to_context === true && (
          <span className="text-green-400">✓ Relevant to this dispute</span>
        )}
        {analysis.is_relevant_to_context === false && (
          <span className="text-yellow-400">⚠ Relevance unclear</span>
        )}
        {analysis.contains_text === true && (
          <span className="text-blue-400">📄 Contains readable text</span>
        )}
      </div>

      {/* Potential issues */}
      {analysis.potential_issues?.filter(i => i !== 'ai_unavailable').length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {analysis.potential_issues
            .filter(i => i !== 'ai_unavailable')
            .map(issue => (
              <span key={issue}
                className="text-xs bg-yellow-900/30 text-yellow-400 px-2 py-0.5 rounded-full">
                ⚠ {issue.replace(/_/g, ' ')}
              </span>
            ))}
        </div>
      )}
    </div>
  );
}