// FILE: frontend/src/pages/Landing.js

import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function Landing() {
  const { login, user } = useAuth();
  const navigate = useNavigate();
  const [form, setForm]     = useState({ email: '', password: '' });
  const [loading, setLoading] = useState(false);
  const [error, setError]   = useState('');

  if (user) {
    navigate('/my-trades');
    return null;
  }

  const handleLogin = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    try {
      await login(form.email, form.password);
      navigate('/my-trades');
    } catch {
      setError('Invalid email or password.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-surface flex flex-col">
      {/* Nav */}
      <nav className="px-8 py-5 flex items-center justify-between border-b border-border">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-brand-500 flex items-center justify-center">
            <span className="text-white font-bold text-sm">AF</span>
          </div>
          <span className="font-bold text-white text-xl">AfriFlow</span>
        </div>
        <Link to="/register"
          className="text-sm text-brand-400 hover:text-brand-300 font-medium transition-colors">
          Create Trade Profile →
        </Link>
      </nav>

      <div className="flex-1 flex flex-col lg:flex-row">
        {/* Left — Hero */}
        <div className="flex-1 flex flex-col justify-center px-8 lg:px-20 py-16">
          <div className="inline-flex items-center gap-2 bg-brand-500/10 border border-brand-500/20 rounded-full px-4 py-1.5 w-fit mb-8">
            <div className="w-1.5 h-1.5 rounded-full bg-brand-500 pulse-soft" />
            <span className="text-brand-400 text-xs font-medium">Pan-African Trade Trust Infrastructure</span>
          </div>

          <h1 className="text-4xl lg:text-6xl font-bold text-white leading-tight mb-6">
            Trade with confidence<br />
            <span className="text-brand-500">across Africa.</span>
          </h1>

          <p className="text-gray-400 text-lg max-w-xl leading-relaxed mb-12">
            AfriFlow gives your business a verified Trade Profile, smart escrow protection,
            and an AI-powered reputation score that grows with every deal you close.
          </p>

          <div className="flex flex-wrap gap-3 mb-12">
            {[
              { icon: '🔐', label: 'Smart Escrow' },
              { icon: '🤖', label: 'AI Dispute Resolution' },
              { icon: '📊', label: 'Trust Score Engine' },
              { icon: '🌍', label: 'Cross-Border Ready' },
            ].map(f => (
              <div key={f.label}
                className="flex items-center gap-2 bg-panel border border-border rounded-full px-4 py-2 text-sm text-gray-300">
                <span>{f.icon}</span>
                <span>{f.label}</span>
              </div>
            ))}
          </div>

          <div className="flex gap-8">
            {[
              { value: '$145T', label: 'Informal B2B market'  },
              { value: '$120B', label: 'Trade finance gap'    },
              { value: '54',   label: 'African countries'     },
            ].map(s => (
              <div key={s.label}>
                <p className="text-2xl font-bold text-brand-400">{s.value}</p>
                <p className="text-xs text-gray-600">{s.label}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Right — Login */}
        <div className="lg:w-[440px] flex items-center justify-center px-8 py-16">
          <div className="w-full max-w-sm space-y-5">
            <div>
              <h2 className="text-2xl font-bold text-white">Sign in</h2>
              <p className="text-gray-500 text-sm mt-1">Access your Trade Profile and trades.</p>
            </div>

            {error && (
              <div className="bg-red-900/30 border border-red-800 rounded-xl px-4 py-3 text-red-400 text-sm">
                {error}
              </div>
            )}

            <form onSubmit={handleLogin} className="space-y-4">
              <div>
                <label className="label">Email</label>
                <input className="input" type="email" placeholder="you@yourbusiness.com"
                  value={form.email}
                  onChange={e => setForm(f => ({ ...f, email: e.target.value }))}
                  required autoFocus
                />
              </div>
              <div>
                <label className="label">Password</label>
                <input className="input" type="password" placeholder="Your password"
                  value={form.password}
                  onChange={e => setForm(f => ({ ...f, password: e.target.value }))}
                  required
                />
              </div>
              <button type="submit" disabled={loading}
                className="btn-primary w-full text-center py-3.5 text-base disabled:opacity-50">
                {loading ? (
                  <span className="flex items-center justify-center gap-2">
                    <div className="w-2 h-2 bg-white rounded-full dot-bounce" />
                    <div className="w-2 h-2 bg-white rounded-full dot-bounce" />
                    <div className="w-2 h-2 bg-white rounded-full dot-bounce" />
                  </span>
                ) : 'Sign In →'}
              </button>
            </form>

            <div className="text-center space-y-1 pt-2">
              <p className="text-sm text-gray-500">
                New to AfriFlow?{' '}
                <Link to="/register" className="text-brand-400 hover:text-brand-300 font-medium">
                  Create your Trade Profile
                </Link>
              </p>
            </div>

            {/* Trust indicators */}
            <div className="border-t border-border pt-5 space-y-2">
              {[
                '🔒 Your data is encrypted and stored securely',
                '✓ Verified businesses only — ID required',
                '🌍 Active in Nigeria, Ghana, Kenya and more',
              ].map(line => (
                <p key={line} className="text-xs text-gray-600">{line}</p>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}