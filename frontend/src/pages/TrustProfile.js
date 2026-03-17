// FILE: frontend/src/pages/TrustProfile.js

import React, { useEffect, useState } from 'react';
import { useParams, useNavigate, useSearchParams } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import api from '../api/client';
import { ScoreRing, MetricBar, DisputeBadge, CorridorBadge } from '../components/TrustScoreBar';

export default function TrustProfile() {
  const { userId } = useParams();
  const [searchParams] = useSearchParams();
  const { user: currentUser } = useAuth();
  const navigate = useNavigate();

  const [data, setData]       = useState(null);
  const [loading, setLoading] = useState(true);
  const [copied, setCopied]   = useState(false);
  const isWelcome = searchParams.get('welcome') === '1';
  const isOwn     = currentUser?.id === parseInt(userId);

  useEffect(() => {
    api.get(`/profile/${userId}`)
      .then(res => setData(res.data))
      .catch(() => navigate('/my-trades'))
      .finally(() => setLoading(false));
  }, [userId]);

  const copyProfileLink = () => {
    const url = `${window.location.origin}/profile/${userId}`;
    navigator.clipboard.writeText(url).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2500);
    });
  };

  if (loading) return <LoadingScreen />;
  if (!data) return null;

  const { user, trust_score: t, trade_history } = data;

  return (
    <div className="max-w-4xl mx-auto px-6 py-8 space-y-5 fade-in">

      {/* Top bar for unauthenticated visitors */}
      {!currentUser && (
        <div className="flex items-center justify-between py-2 mb-4">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-lg bg-brand-500 flex items-center justify-center">
              <span className="text-white font-bold text-xs">AF</span>
            </div>
            <span className="font-bold text-white">AfriFlow</span>
          </div>
          <div className="flex items-center gap-3">
            <button onClick={() => navigate('/')} className="btn-ghost text-sm">Sign In</button>
            <button onClick={() => navigate('/register')} className="btn-primary text-sm py-2">
              Create Trade Profile
            </button>
          </div>
        </div>
      )}

      {/* Welcome banner — shown once after registration */}
      {isWelcome && isOwn && (
        <div className="card bg-green-900/20 border-green-800/40 space-y-2">
          <div className="flex items-center gap-3">
            <span className="text-2xl">🎉</span>
            <div>
              <p className="font-semibold text-green-300">Your Trade Profile is live!</p>
              <p className="text-sm text-green-500">
                Share your profile link with suppliers and buyers so they can verify you before trading.
              </p>
            </div>
          </div>
          <button onClick={copyProfileLink}
            className="text-sm font-medium text-green-400 hover:text-green-300 transition-colors">
            {copied ? '✓ Link copied!' : 'Copy your profile link →'}
          </button>
        </div>
      )}

      {/* Header Card */}
      <div className="card">
        <div className="flex flex-col md:flex-row gap-6 items-start">
          <div className="flex items-start gap-4 flex-1">
            {/* Avatar */}
            <div className="w-16 h-16 rounded-2xl bg-brand-500/20 flex items-center justify-center text-brand-400 font-bold text-2xl flex-shrink-0">
              {user.avatar_initials}
            </div>

            <div className="flex-1 min-w-0">
              {/* Business name + badges */}
              <div className="flex items-center gap-2 flex-wrap">
                <h1 className="text-2xl font-bold text-white">{user.business_name}</h1>
                {user.verified && (
                  <span className="bg-green-900/40 text-green-400 text-xs px-2 py-0.5 rounded-full font-medium">
                    ✓ Verified
                  </span>
                )}
                {user.has_reg_document && (
                  <span className="bg-blue-900/40 text-blue-400 text-xs px-2 py-0.5 rounded-full font-medium">
                    ✓ Registered
                  </span>
                )}
                {user.verification_status === 'pending' && (
                  <span className="bg-yellow-900/40 text-yellow-400 text-xs px-2 py-0.5 rounded-full">
                    Verification Pending
                  </span>
                )}
              </div>

              {/* Owner name */}
              <p className="text-gray-400 text-sm mt-0.5">{user.name}</p>

              {/* Location */}
              <div className="flex items-center gap-1.5 mt-1 text-sm text-gray-500">
                <span>📍</span>
                <span>{user.city ? `${user.city}, ` : ''}{user.location}, {user.country}</span>
              </div>

              {/* Business type */}
              <p className="text-xs text-gray-600 mt-1">🏢 {user.business_type}</p>

              {/* Contact */}
              <div className="flex flex-wrap gap-3 mt-2">
                {user.phone && (
                  <span className="text-xs text-gray-500 flex items-center gap-1">
                    📞 {user.phone}
                  </span>
                )}
                {user.whatsapp && (
                  <span className="text-xs text-gray-500 flex items-center gap-1">
                    💬 WA: {user.whatsapp}
                  </span>
                )}
                {user.website && (
                  <a href={user.website} target="_blank" rel="noopener noreferrer"
                    className="text-xs text-brand-400 hover:text-brand-300 transition-colors">
                    🌐 {user.website.replace(/^https?:\/\//, '')}
                  </a>
                )}
              </div>

              {/* Trade ID */}
              <div className="flex items-center gap-2 mt-3">
                <span className="font-mono text-xs text-brand-400 bg-brand-900/20 px-2 py-0.5 rounded">
                  {user.trade_id}
                </span>
                <span className="text-xs text-gray-600">
                  Member since {new Date(user.created_at).toLocaleDateString('en-GB', { month: 'short', year: 'numeric' })}
                </span>
              </div>
            </div>
          </div>

          {/* Score ring */}
          <div className="flex flex-col items-center gap-2">
            <ScoreRing score={t.overall_score} size={110} />
            <p className="text-xs text-gray-500">Trust Score</p>
            {t.overall_score !== t.previous_score && (
              <div className={`text-xs font-medium ${t.overall_score > t.previous_score ? 'text-green-400' : 'text-red-400'}`}>
                {t.overall_score > t.previous_score ? '↑' : '↓'} from {Math.round(t.previous_score)}
              </div>
            )}
          </div>
        </div>

        {/* Products traded */}
        {user.products_traded?.length > 0 && (
          <div className="mt-5 pt-4 border-t border-border">
            <p className="text-xs text-gray-500 uppercase tracking-wide mb-2">Products & Services</p>
            <div className="flex flex-wrap gap-2">
              {user.products_traded.map(p => (
                <span key={p} className="text-xs bg-border text-gray-300 px-3 py-1.5 rounded-full">
                  {p}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Share button — shown on own profile */}
        {isOwn && (
          <div className="mt-4 pt-4 border-t border-border flex items-center justify-between">
            <p className="text-xs text-gray-500">Share your profile with potential trade partners</p>
            <button onClick={copyProfileLink}
              className={`text-xs font-medium px-4 py-2 rounded-lg transition-all
                ${copied
                  ? 'bg-green-900/30 text-green-400'
                  : 'bg-border text-gray-400 hover:text-white hover:bg-gray-700'}`}>
              {copied ? '✓ Copied!' : '🔗 Copy Profile Link'}
            </button>
          </div>
        )}
      </div>

      {/* Score Breakdown */}
      <div className="card space-y-5">
        <h2 className="font-semibold text-white">Trust Score Breakdown</h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <MetricBar label="Payment Reliability" value={t.payment_reliability} />
            <MetricBar label="Delivery Accuracy"   value={t.delivery_accuracy}   />
          </div>
          <div className="space-y-4">
            <div>
              <p className="text-sm text-gray-400 mb-2">Dispute Rate</p>
              <DisputeBadge label={t.dispute_rate} />
            </div>
            <div>
              <p className="text-sm text-gray-400 mb-2">Corridor Experience</p>
              <CorridorBadge label={t.corridor_experience} />
            </div>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-4 pt-4 border-t border-border">
          <Stat value={t.total_trades}   label="Completed Trades" />
          <Stat value={t.total_disputes} label="Disputes Raised"  />
          <Stat value={t.disputes_won}   label="Disputes Won"     />
        </div>
      </div>

      {/* AI Scoring Intelligence */}
      {(t.score_reasoning || t.score_trajectory) && (
        <div className="card space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="font-semibold text-white">AI Scoring Intelligence</h2>
            <span className="text-xs bg-brand-900/30 text-brand-400 border border-brand-800/30 px-2 py-0.5 rounded-full">
              🤖 llama-3.3-70b-versatile
            </span>
          </div>
          {t.score_trajectory && (
            <div className="flex items-center gap-3">
              <span className="text-xs text-gray-500">Score Trajectory</span>
              <TrajectoryBadge trajectory={t.score_trajectory} />
            </div>
          )}
          {t.score_reasoning && (
            <div className="bg-surface rounded-xl p-4">
              <p className="text-xs text-gray-500 mb-2">Why this score</p>
              <p className="text-sm text-gray-300 leading-relaxed">{t.score_reasoning}</p>
            </div>
          )}
          {t.score_risk_flag && (
            <div className="bg-yellow-900/10 border border-yellow-800/30 rounded-xl p-3 flex items-start gap-2">
              <span className="text-yellow-400">⚠</span>
              <p className="text-sm text-yellow-300">{t.score_risk_flag}</p>
            </div>
          )}
          <p className="text-xs text-gray-700">
            Updated after every completed trade ·
            {t.score_source === 'groq_ai_scoring' ? ' AI pattern analysis' : ' Deterministic engine'}
          </p>
        </div>
      )}

      {/* Trade History */}
      {trade_history?.length > 0 && (
        <div className="card space-y-4">
          <h2 className="font-semibold text-white">Trade History</h2>
          <div className="space-y-3">
            {trade_history.map(event => (
              <HistoryRow key={event.id} event={event} />
            ))}
          </div>
        </div>
      )}

      {/* CTA — behaviour depends on auth state */}
      {!isOwn && (
        currentUser ? (
          <div className="flex gap-3">
            <button
              onClick={() => navigate(`/trade/create?supplier=${userId}`)}
              className="btn-primary flex-1 text-center">
              Start a Trade with {user.business_name} →
            </button>
          </div>
        ) : (
          <div className="card text-center space-y-3">
            <p className="text-gray-400 text-sm">
              Create your own Trade Profile to start trading with <span className="text-white font-medium">{user.business_name}</span>.
            </p>
            <div className="flex gap-3 justify-center">
              <button onClick={() => navigate('/register')} className="btn-primary">
                Create Trade Profile →
              </button>
              <button onClick={() => navigate('/')} className="btn-secondary">
                Sign In
              </button>
            </div>
          </div>
        )
      )}
    </div>
  );
}

// ── Sub-components ─────────────────────────────────────────────

function TrajectoryBadge({ trajectory }) {
  const config = {
    improving:         { label: '↑ Improving',        color: 'text-green-400', bg: 'bg-green-900/30'  },
    stable:            { label: '→ Stable',            color: 'text-blue-400',  bg: 'bg-blue-900/30'   },
    declining:         { label: '↓ Declining',         color: 'text-red-400',   bg: 'bg-red-900/30'    },
    insufficient_data: { label: '· No data yet',       color: 'text-gray-400',  bg: 'bg-gray-800'      },
  };
  const c = config[trajectory] || config.insufficient_data;
  return (
    <span className={`status-badge text-xs ${c.bg} ${c.color}`}>{c.label}</span>
  );
}

function Stat({ value, label }) {
  return (
    <div className="text-center">
      <p className="text-2xl font-bold text-white">{value}</p>
      <p className="text-xs text-gray-500 mt-0.5">{label}</p>
    </div>
  );
}

function HistoryRow({ event }) {
  const icons = {
    trade_created:         '📝',
    trade_accepted:        '✅',
    escrow_funded:         '🔐',
    goods_shipped:         '📦',
    delivery_confirmed:    '✓',
    funds_released:        '💰',
    dispute_raised:        '⚠️',
    dispute_resolved:      '⚖️',
    trust_updated:         '📊',
    trade_settled:         '🏆',
    dispute_won:           '🏅',
  };
  const date = new Date(event.created_at).toLocaleDateString('en-GB', {
    day: 'numeric', month: 'short', year: 'numeric'
  });
  return (
    <div className="flex items-start gap-3">
      <span className="text-sm mt-0.5">{icons[event.event_type] || '•'}</span>
      <div className="flex-1">
        <p className="text-sm text-gray-300">{event.description}</p>
        <p className="text-xs text-gray-600 mt-0.5">{date}</p>
      </div>
    </div>
  );
}

function LoadingScreen() {
  return (
    <div className="max-w-4xl mx-auto px-6 py-8 space-y-4">
      {[1, 2, 3].map(i => (
        <div key={i} className="card h-32 animate-pulse bg-panel" />
      ))}
    </div>
  );
}