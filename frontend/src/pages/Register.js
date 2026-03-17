// FILE: frontend/src/pages/Register.js
//
// 3-step Trade Profile registration:
//   Step 1 — Account credentials (name, email, password)
//   Step 2 — Business identity (business name, type, products, location)
//   Step 3 — Verification (owner ID upload, business registration, phone)

import React, { useState, useRef } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import api from '../api/client';

// ── IDVerificationResult component ───────────────────────────

function IDVerificationResult({ result }) {
  if (!result) return null;

  const passed  = result.passed;
  const confidence = result.confidence || 0;

  return (
    <div className={`rounded-xl border p-4 space-y-3 ${
      passed
        ? 'bg-green-900/15 border-green-800/40'
        : 'bg-red-900/15 border-red-800/40'
    }`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-lg">{passed ? '✅' : '❌'}</span>
          <span className={`font-semibold text-sm ${passed ? 'text-green-300' : 'text-red-300'}`}>
            {passed ? 'ID Verified' : 'Verification Failed'}
          </span>
          <span className="text-xs text-purple-400 bg-purple-900/30 px-2 py-0.5 rounded-full">
            🤖 AI
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className={`h-2 w-16 rounded-full overflow-hidden bg-border`}>
            <div
              className={`h-full rounded-full ${confidence >= 70 ? 'bg-green-500' : confidence >= 50 ? 'bg-yellow-500' : 'bg-red-500'}`}
              style={{ width: `${confidence}%` }}
            />
          </div>
          <span className={`text-xs font-medium ${confidence >= 70 ? 'text-green-400' : confidence >= 50 ? 'text-yellow-400' : 'text-red-400'}`}>
            {confidence}%
          </span>
        </div>
      </div>

      {/* Failure reason */}
      {!passed && result.rejection_reason && (
        <p className="text-sm text-red-300">{result.rejection_reason}</p>
      )}

      {/* Extracted name */}
      {result.name_on_document && (
        <div className="flex items-center gap-2 text-sm">
          <span className="text-gray-500">Name on document:</span>
          <span className="text-white font-medium">{result.name_on_document}</span>
        </div>
      )}

      {/* Details row */}
      <div className="flex flex-wrap gap-3 text-xs">
        {result.image_quality && (
          <span className={`${result.image_quality === 'clear' ? 'text-green-400' : result.image_quality === 'acceptable' ? 'text-yellow-400' : 'text-red-400'}`}>
            📷 Image: {result.image_quality}
          </span>
        )}
        {result.declared_type_matches === true && (
          <span className="text-green-400">✓ Type matches</span>
        )}
        {result.declared_type_matches === false && (
          <span className="text-yellow-400">⚠ Type mismatch</span>
        )}
        {result.appears_authentic === true && (
          <span className="text-green-400">✓ Appears authentic</span>
        )}
        {result.appears_authentic === false && (
          <span className="text-red-400">✗ Authenticity concern</span>
        )}
        {result.country_detected && (
          <span className="text-gray-400">🌍 {result.country_detected}</span>
        )}
      </div>

      {/* Flags */}
      {result.flags?.filter(f => f !== 'ai_unavailable').length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {result.flags.filter(f => f !== 'ai_unavailable').map(flag => (
            <span key={flag} className="text-xs bg-red-900/30 text-red-400 px-2 py-0.5 rounded-full">
              {flag.replace(/_/g, ' ')}
            </span>
          ))}
        </div>
      )}

      {/* Retry prompt on failure */}
      {!passed && (
        <p className="text-xs text-gray-500">
          Take a new photo in good lighting and ensure the full document is visible.
          You can still submit — your ID will be reviewed manually if AI verification fails.
        </p>
      )}
    </div>
  );
}


const BUSINESS_TYPES = [
  'Textile & Fashion',
  'Food & Agriculture',
  'Electronics & Technology',
  'Furniture & Home Goods',
  'Building & Construction',
  'Health & Pharmaceuticals',
  'Automotive & Parts',
  'Beauty & Personal Care',
  'Arts & Crafts',
  'Logistics & Supply Chain',
  'Manufacturing',
  'General Trade',
  'Other',
];

const AFRICAN_COUNTRIES = [
  'Nigeria', 'Ghana', 'Kenya', 'South Africa', 'Ethiopia', 'Tanzania',
  'Uganda', 'Rwanda', 'Senegal', "Côte d'Ivoire", 'Cameroon', 'Egypt',
  'Morocco', 'Zimbabwe', 'Zambia', 'Mozambique', 'Angola', 'Botswana',
  'Namibia', 'Mali', 'Burkina Faso', 'Niger', 'Chad', 'Sudan',
  'Somalia', 'Eritrea', 'Djibouti', 'Comoros', 'Mauritius', 'Seychelles',
  'Madagascar', 'Malawi', 'Lesotho', 'Eswatini', 'Benin', 'Togo',
  'Guinea', 'Sierra Leone', 'Liberia', 'Gambia', 'Guinea-Bissau',
  'Cape Verde', 'São Tomé and Príncipe', 'Gabon', 'Equatorial Guinea',
  'Republic of Congo', 'DR Congo', 'Central African Republic',
  'Libya', 'Tunisia', 'Algeria', 'Mauritania',
];

const ID_TYPES = [
  { value: 'national_id',       label: 'National ID Card'       },
  { value: 'passport',          label: 'International Passport' },
  { value: 'drivers_license',   label: "Driver's Licence"       },
  { value: 'voters_card',       label: "Voter's Card"           },
];

const PRODUCT_SUGGESTIONS = [
  'Textiles', 'Ankara Fabric', 'Kente', 'Lace Fabric', 'Wax Print',
  'Clothing & Apparel', 'Fashion Accessories',
  'Rice', 'Maize', 'Palm Oil', 'Cassava', 'Groundnuts', 'Coffee', 'Cocoa',
  'Electronics', 'Mobile Phones', 'Computers',
  'Timber', 'Furniture', 'Building Materials', 'Steel',
  'Pharmaceuticals', 'Medical Supplies',
  'Cosmetics', 'Hair Products', 'Skincare',
];

// ── Step indicators ────────────────────────────────────────────

function StepIndicator({ step, total }) {
  return (
    <div className="flex items-center gap-0 mb-8">
      {Array.from({ length: total }, (_, i) => {
        const n = i + 1;
        const done    = n < step;
        const current = n === step;
        return (
          <React.Fragment key={n}>
            <div className={`flex flex-col items-center`}>
              <div className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold transition-all
                ${done    ? 'bg-green-500 text-white'  :
                  current ? 'bg-brand-500 text-white'  :
                            'bg-border text-gray-600'}`}>
                {done ? '✓' : n}
              </div>
              <span className={`text-xs mt-1 hidden sm:block
                ${current ? 'text-white' : 'text-gray-600'}`}>
                {n === 1 ? 'Account' : n === 2 ? 'Business' : 'Verify'}
              </span>
            </div>
            {i < total - 1 && (
              <div className={`flex-1 h-px mx-2 mb-4 transition-colors
                ${done ? 'bg-green-500' : 'bg-border'}`} />
            )}
          </React.Fragment>
        );
      })}
    </div>
  );
}

// ── Main component ─────────────────────────────────────────────

export default function Register() {
  const { login } = useAuth();
  const navigate   = useNavigate();

  const [step, setStep]     = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError]   = useState('');

  // Step 1
  const [account, setAccount] = useState({ name: '', email: '', password: '', confirm: '' });
  // Step 2
  const [business, setBusiness] = useState({
    business_name: '', business_type: '', products: [], customProduct: '',
    location: '', city: '', country: 'Nigeria', website: '',
  });
  // Step 3
  const [verify, setVerify] = useState({
    phone: '', whatsapp: '',
    id_document_type: '', id_file: null, id_preview: null,
    registration_number: '', reg_file: null,
  });

  const idFileRef  = useRef(null);
  const regFileRef = useRef(null);

  // ID verification state
  const [idVerifying, setIdVerifying]   = useState(false);
  const [idVerification, setIdVerification] = useState(null); // result from AI

  // ── Validation ───────────────────────────────────────────────

  const validateStep1 = () => {
    if (!account.name.trim())    return 'Full name is required.';
    if (!account.email.trim())   return 'Email is required.';
    if (!/\S+@\S+\.\S+/.test(account.email)) return 'Enter a valid email address.';
    if (account.password.length < 8) return 'Password must be at least 8 characters.';
    if (account.password !== account.confirm) return 'Passwords do not match.';
    return null;
  };

  const validateStep2 = () => {
    if (!business.business_name.trim()) return 'Business name is required.';
    if (!business.business_type)        return 'Select a business type.';
    if (!business.location.trim())      return 'Location is required.';
    if (!business.country)              return 'Select your country.';
    if (business.products.length === 0) return 'Add at least one product or service you trade.';
    return null;
  };

  const validateStep3 = () => {
    if (!verify.phone.trim())         return 'Phone number is required.';
    if (!verify.id_document_type)     return 'Select your ID document type.';
    if (!verify.id_file)              return 'Upload your ID document photo.';
    return null;
  };

  // ── Navigation ───────────────────────────────────────────────

  const nextStep = () => {
    setError('');
    const err = step === 1 ? validateStep1() : step === 2 ? validateStep2() : null;
    if (err) { setError(err); return; }
    setStep(s => s + 1);
  };

  // ── Product tag management ────────────────────────────────────

  const addProduct = (p) => {
    const trimmed = p.trim();
    if (!trimmed) return;
    if (!business.products.includes(trimmed)) {
      setBusiness(b => ({ ...b, products: [...b.products, trimmed], customProduct: '' }));
    }
  };

  const removeProduct = (p) => {
    setBusiness(b => ({ ...b, products: b.products.filter(x => x !== p) }));
  };

  // ── File handlers ─────────────────────────────────────────────

  const handleIdFile = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const preview = URL.createObjectURL(file);
    setVerify(v => ({ ...v, id_file: file, id_preview: preview }));
    setIdVerification(null);

    // ── Run AI verification immediately ────────
    if (!verify.id_document_type) {
      setIdVerification({
        skipped: true,
        message: 'Select your ID type first, then re-upload for instant AI verification.'
      });
      return;
    }

    setIdVerifying(true);
    try {
      const form = new FormData();
      form.append('file',          file);
      form.append('document_type', verify.id_document_type);
      form.append('owner_name',    account.name || 'Unknown');

      const res = await api.post('/verify/id', form, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setIdVerification(res.data.verification);
    } catch (err) {
      // Network/server error — don't block registration
      setIdVerification({
        error:   true,
        message: 'Could not reach verification service. Your ID will be reviewed manually.'
      });
    } finally {
      setIdVerifying(false);
    }
  };

  const handleRegFile = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setVerify(v => ({ ...v, reg_file: file }));
  };

  // ── Submit ────────────────────────────────────────────────────

  const handleSubmit = async () => {
    const err = validateStep3();
    if (err) { setError(err); return; }

    setLoading(true);
    setError('');

    try {
      const form = new FormData();

      // Step 1
      form.append('name',     account.name.trim());
      form.append('email',    account.email.trim().toLowerCase());
      form.append('password', account.password);

      // Step 2
      form.append('business_name', business.business_name.trim());
      form.append('business_type', business.business_type);
      form.append('products_traded', business.products.join(', '));
      form.append('location',  business.location.trim());
      form.append('city',      business.city.trim());
      form.append('country',   business.country);
      if (business.website) form.append('website', business.website.trim());

      // Step 3
      form.append('phone',            verify.phone.trim());
      form.append('whatsapp',         verify.whatsapp.trim());
      form.append('id_document_type', verify.id_document_type);
      if (verify.registration_number)
        form.append('registration_number', verify.registration_number.trim());
      if (verify.id_file)  form.append('id_document',  verify.id_file);
      if (verify.reg_file) form.append('reg_document', verify.reg_file);

      const res = await api.post('/auth/register', form, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      // Auto-login
      const { token, user } = res.data;
      localStorage.setItem('afriflow_token', token);
      api.defaults.headers.common['Authorization'] = `Bearer ${token}`;

      navigate(`/profile/${user.id}?welcome=1`);
    } catch (e) {
      setError(e.response?.data?.error || 'Registration failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // ── Render ────────────────────────────────────────────────────

  return (
    <div className="min-h-screen bg-surface flex flex-col">
      {/* Navbar */}
      <nav className="px-6 py-4 flex items-center justify-between border-b border-border">
        <Link to="/" className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-brand-500 flex items-center justify-center">
            <span className="text-white font-bold text-sm">AF</span>
          </div>
          <span className="font-bold text-white text-lg">AfriFlow</span>
        </Link>
        <p className="text-sm text-gray-500">
          Already have an account?{' '}
          <Link to="/" className="text-brand-400 hover:text-brand-300 font-medium">
            Sign in
          </Link>
        </p>
      </nav>

      <div className="flex-1 flex items-start justify-center px-4 py-10">
        <div className="w-full max-w-lg">
          {/* Header */}
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-white">Create your Trade Profile</h1>
            <p className="text-gray-500 mt-1.5 text-sm">
              Your Trade Profile is your verified digital business identity on AfriFlow.
              Other businesses see it before agreeing to trade with you.
            </p>
          </div>

          <StepIndicator step={step} total={3} />

          {/* Error */}
          {error && (
            <div className="bg-red-900/30 border border-red-800 rounded-xl px-4 py-3 text-red-400 text-sm mb-5">
              {error}
            </div>
          )}

          {/* ── STEP 1: Account ─────────────────────── */}
          {step === 1 && (
            <div className="card space-y-4 fade-in">
              <div className="mb-1">
                <h2 className="font-semibold text-white text-lg">Your account</h2>
                <p className="text-xs text-gray-500 mt-0.5">This is your personal login — not your public business profile.</p>
              </div>

              <div>
                <label className="label">Full Name</label>
                <input className="input" placeholder="Amaka Okonkwo"
                  value={account.name}
                  onChange={e => setAccount(a => ({ ...a, name: e.target.value }))}
                />
              </div>
              <div>
                <label className="label">Email Address</label>
                <input className="input" type="email" placeholder="amaka@yourbusiness.com"
                  value={account.email}
                  onChange={e => setAccount(a => ({ ...a, email: e.target.value }))}
                />
              </div>
              <div>
                <label className="label">Password</label>
                <input className="input" type="password" placeholder="Minimum 8 characters"
                  value={account.password}
                  onChange={e => setAccount(a => ({ ...a, password: e.target.value }))}
                />
              </div>
              <div>
                <label className="label">Confirm Password</label>
                <input className="input" type="password" placeholder="Re-enter your password"
                  value={account.confirm}
                  onChange={e => setAccount(a => ({ ...a, confirm: e.target.value }))}
                />
              </div>

              <button onClick={nextStep} className="btn-primary w-full text-center mt-2">
                Continue to Business Details →
              </button>
            </div>
          )}

          {/* ── STEP 2: Business ────────────────────── */}
          {step === 2 && (
            <div className="card space-y-4 fade-in">
              <div className="mb-1">
                <h2 className="font-semibold text-white text-lg">Your business</h2>
                <p className="text-xs text-gray-500 mt-0.5">This is your public Trade Profile — what other businesses will see.</p>
              </div>

              <div>
                <label className="label">Business Name</label>
                <input className="input" placeholder="e.g. Amaka Clothing Co."
                  value={business.business_name}
                  onChange={e => setBusiness(b => ({ ...b, business_name: e.target.value }))}
                />
              </div>

              <div>
                <label className="label">Business Type</label>
                <select className="input"
                  value={business.business_type}
                  onChange={e => setBusiness(b => ({ ...b, business_type: e.target.value }))}>
                  <option value="">Select a category...</option>
                  {BUSINESS_TYPES.map(t => <option key={t} value={t}>{t}</option>)}
                </select>
              </div>

              <div>
                <label className="label">What do you trade? <span className="text-gray-600 font-normal">(add at least one)</span></label>
                {/* Tag chips */}
                {business.products.length > 0 && (
                  <div className="flex flex-wrap gap-2 mb-2">
                    {business.products.map(p => (
                      <span key={p}
                        className="flex items-center gap-1 bg-brand-900/30 text-brand-400 text-xs px-3 py-1.5 rounded-full border border-brand-800/30">
                        {p}
                        <button type="button" onClick={() => removeProduct(p)}
                          className="text-brand-600 hover:text-red-400 ml-0.5 transition-colors">✕</button>
                      </span>
                    ))}
                  </div>
                )}
                {/* Suggestion chips */}
                <div className="flex flex-wrap gap-1.5 mb-2">
                  {PRODUCT_SUGGESTIONS.filter(s => !business.products.includes(s)).slice(0, 10).map(s => (
                    <button key={s} type="button" onClick={() => addProduct(s)}
                      className="text-xs px-2.5 py-1 rounded-full border border-border text-gray-500 hover:border-brand-500 hover:text-brand-400 transition-all">
                      + {s}
                    </button>
                  ))}
                </div>
                {/* Custom input */}
                <div className="flex gap-2">
                  <input className="input text-sm flex-1" placeholder="Type a product and press Enter..."
                    value={business.customProduct}
                    onChange={e => setBusiness(b => ({ ...b, customProduct: e.target.value }))}
                    onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); addProduct(business.customProduct); }}}
                  />
                  <button type="button" onClick={() => addProduct(business.customProduct)}
                    className="btn-secondary text-sm px-4">Add</button>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="label">Country</label>
                  <select className="input"
                    value={business.country}
                    onChange={e => setBusiness(b => ({ ...b, country: e.target.value }))}>
                    {AFRICAN_COUNTRIES.map(c => <option key={c} value={c}>{c}</option>)}
                  </select>
                </div>
                <div>
                  <label className="label">City</label>
                  <input className="input" placeholder="Lagos"
                    value={business.city}
                    onChange={e => setBusiness(b => ({ ...b, city: e.target.value }))}
                  />
                </div>
              </div>

              <div>
                <label className="label">Full Address / Area</label>
                <input className="input" placeholder="e.g. Balogun Market, Lagos Island"
                  value={business.location}
                  onChange={e => setBusiness(b => ({ ...b, location: e.target.value }))}
                />
              </div>

              <div>
                <label className="label">Website <span className="text-gray-600 font-normal">(optional)</span></label>
                <input className="input" placeholder="https://yourbusiness.com"
                  value={business.website}
                  onChange={e => setBusiness(b => ({ ...b, website: e.target.value }))}
                />
              </div>

              <div className="flex gap-3 pt-2">
                <button onClick={() => setStep(1)} className="btn-secondary flex-1 text-center">← Back</button>
                <button onClick={nextStep} className="btn-primary flex-1 text-center">
                  Continue to Verification →
                </button>
              </div>
            </div>
          )}

          {/* ── STEP 3: Verification ────────────────── */}
          {step === 3 && (
            <div className="space-y-4 fade-in">
              {/* What this is */}
              <div className="card bg-brand-900/10 border-brand-800/30 space-y-1">
                <p className="text-sm font-medium text-brand-300">Why we verify</p>
                <p className="text-xs text-gray-400 leading-relaxed">
                  Your Trade Profile is shared with other businesses before they trade with you.
                  Verification builds trust with every counterparty — they can see you are real.
                </p>
              </div>

              <div className="card space-y-4">
                <h2 className="font-semibold text-white text-lg">Phone & Contact</h2>

                <div>
                  <label className="label">Phone Number <span className="text-red-400">*</span></label>
                  <input className="input" placeholder="+234 801 234 5678"
                    value={verify.phone}
                    onChange={e => setVerify(v => ({ ...v, phone: e.target.value }))}
                  />
                </div>
                <div>
                  <label className="label">WhatsApp Number <span className="text-gray-600 font-normal">(if different)</span></label>
                  <input className="input" placeholder="+234 801 234 5678"
                    value={verify.whatsapp}
                    onChange={e => setVerify(v => ({ ...v, whatsapp: e.target.value }))}
                  />
                </div>
              </div>

              {/* ID Document */}
              <div className="card space-y-4">
                <div>
                  <h2 className="font-semibold text-white text-lg">Owner ID</h2>
                  <p className="text-xs text-gray-500 mt-0.5">A government-issued ID for the business owner. Not shown to counterparties — used for verification only.</p>
                </div>

                <div>
                  <label className="label">ID Type <span className="text-red-400">*</span></label>
                  <select className="input"
                    value={verify.id_document_type}
                    onChange={e => {
                      setVerify(v => ({ ...v, id_document_type: e.target.value }));
                      // Clear result if type changes — needs re-verification
                      if (verify.id_file) setIdVerification(null);
                    }}>
                    <option value="">Select document type...</option>
                    {ID_TYPES.map(t => <option key={t.value} value={t.value}>{t.label}</option>)}
                  </select>
                </div>

                <div>
                  <label className="label">Upload ID Document <span className="text-red-400">*</span></label>
                  <input type="file" ref={idFileRef} accept="image/jpeg,image/png,image/gif,image/webp"
                    onChange={handleIdFile} className="hidden" />

                  {/* Upload button */}
                  {!verify.id_preview && (
                    <button type="button" onClick={() => idFileRef.current?.click()}
                      disabled={!verify.id_document_type}
                      className="w-full border-2 border-dashed border-border hover:border-purple-600 disabled:opacity-40 disabled:cursor-not-allowed rounded-xl p-8 text-center transition-colors group">
                      <div className="text-3xl mb-2 group-hover:scale-110 transition-transform">🪪</div>
                      <p className="text-sm text-gray-400">Click to upload your ID</p>
                      <p className="text-xs text-gray-600 mt-1">JPEG or PNG · Analyzed by llama-3.2-11b-vision-preview</p>
                      {!verify.id_document_type && (
                        <p className="text-xs text-yellow-500 mt-2">← Select ID type first</p>
                      )}
                    </button>
                  )}

                  {/* Preview + AI verification result */}
                  {verify.id_preview && (
                    <div className="space-y-3">
                      {/* Image preview */}
                      <div className="relative">
                        <img src={verify.id_preview} alt="ID document"
                          className="w-full max-h-44 object-cover rounded-xl border border-border" />
                        <button type="button"
                          onClick={() => {
                            setVerify(v => ({ ...v, id_file: null, id_preview: null }));
                            setIdVerification(null);
                            if (idFileRef.current) idFileRef.current.value = '';
                          }}
                          className="absolute top-2 right-2 bg-black/60 text-white text-xs px-2 py-1 rounded-lg hover:bg-red-900/80 transition-colors">
                          Remove
                        </button>
                      </div>

                      {/* AI verifying spinner */}
                      {idVerifying && (
                        <div className="flex items-center gap-3 bg-purple-900/20 border border-purple-800/30 rounded-xl p-3">
                          <div className="flex gap-1">
                            <div className="w-1.5 h-1.5 bg-purple-400 rounded-full dot-bounce" />
                            <div className="w-1.5 h-1.5 bg-purple-400 rounded-full dot-bounce" />
                            <div className="w-1.5 h-1.5 bg-purple-400 rounded-full dot-bounce" />
                          </div>
                          <p className="text-sm text-purple-300">
                            AI verifying your ID document...
                          </p>
                          <span className="text-xs text-purple-500 ml-auto">llama-3.2-11b-vision-preview</span>
                        </div>
                      )}

                      {/* AI verification result */}
                      {!idVerifying && idVerification && !idVerification.skipped && !idVerification.error && (
                        <IDVerificationResult result={idVerification} />
                      )}

                      {/* Skipped / select type first */}
                      {idVerification?.skipped && (
                        <div className="bg-yellow-900/20 border border-yellow-800/30 rounded-xl p-3">
                          <p className="text-xs text-yellow-400">{idVerification.message}</p>
                          <button type="button" onClick={() => idFileRef.current?.click()}
                            className="text-xs text-yellow-300 underline mt-1">
                            Re-upload after selecting type
                          </button>
                        </div>
                      )}

                      {/* API error */}
                      {idVerification?.error && (
                        <div className="bg-gray-800 border border-border rounded-xl p-3">
                          <p className="text-xs text-gray-400">{idVerification.message}</p>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>

              {/* Business Registration (optional) */}
              <div className="card space-y-4">
                <div>
                  <h2 className="font-semibold text-white">Business Registration <span className="text-gray-500 font-normal text-sm">(optional)</span></h2>
                  <p className="text-xs text-gray-500 mt-0.5">CAC (Nigeria), BRELA (Tanzania), or equivalent. Adds a ✓ Registered badge to your profile.</p>
                </div>

                <div>
                  <label className="label">Registration Number</label>
                  <input className="input" placeholder="e.g. RC-1234567"
                    value={verify.registration_number}
                    onChange={e => setVerify(v => ({ ...v, registration_number: e.target.value }))}
                  />
                </div>

                <div>
                  <label className="label">Upload Registration Document</label>
                  <input type="file" ref={regFileRef} accept="image/*,.pdf"
                    onChange={handleRegFile} className="hidden" />
                  {verify.reg_file ? (
                    <div className="bg-surface rounded-xl p-4 flex items-center gap-3 border border-green-800/30">
                      <span className="text-2xl">📄</span>
                      <div>
                        <p className="text-sm text-green-400 font-medium">Uploaded: {verify.reg_file.name}</p>
                        <button type="button" onClick={() => setVerify(v => ({ ...v, reg_file: null }))}
                          className="text-xs text-gray-500 hover:text-red-400 mt-0.5">Remove</button>
                      </div>
                    </div>
                  ) : (
                    <button type="button" onClick={() => regFileRef.current?.click()}
                      className="w-full border border-dashed border-border hover:border-brand-500 rounded-xl py-4 text-center transition-colors text-sm text-gray-500 hover:text-gray-400">
                      📎 Upload CAC certificate or equivalent
                    </button>
                  )}
                </div>
              </div>

              {/* Profile preview */}
              <div className="card bg-surface space-y-3">
                <p className="text-xs text-gray-500 uppercase tracking-wide">Your Trade Profile Preview</p>
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 rounded-full bg-brand-500/20 flex items-center justify-center text-brand-400 font-bold text-lg">
                    {business.business_name?.slice(0, 2).toUpperCase() || '?'}
                  </div>
                  <div>
                    <p className="font-semibold text-white">{business.business_name || 'Your Business'}</p>
                    <p className="text-xs text-gray-500">{business.city ? `${business.city}, ` : ''}{business.country}</p>
                  </div>
                  {(verify.id_file || verify.reg_file) && (
                    <span className="ml-auto text-xs bg-green-900/40 text-green-400 px-2 py-0.5 rounded-full">✓ Verified</span>
                  )}
                </div>
                {business.products.length > 0 && (
                  <div className="flex flex-wrap gap-1.5">
                    {business.products.slice(0, 4).map(p => (
                      <span key={p} className="text-xs bg-border text-gray-400 px-2 py-1 rounded-full">{p}</span>
                    ))}
                    {business.products.length > 4 && (
                      <span className="text-xs text-gray-600">+{business.products.length - 4} more</span>
                    )}
                  </div>
                )}
              </div>

              <div className="flex gap-3">
                <button onClick={() => { setStep(2); setError(''); }} className="btn-secondary flex-1 text-center">← Back</button>
                <button onClick={handleSubmit} disabled={loading} className="btn-primary flex-1 text-center disabled:opacity-50">
                  {loading ? (
                    <span className="flex items-center justify-center gap-2">
                      <div className="w-2 h-2 bg-white rounded-full dot-bounce" />
                      <div className="w-2 h-2 bg-white rounded-full dot-bounce" />
                      <div className="w-2 h-2 bg-white rounded-full dot-bounce" />
                    </span>
                  ) : 'Create Trade Profile →'}
                </button>
              </div>

              <p className="text-center text-xs text-gray-600">
                Your ID is stored securely and never shared with other businesses.
                It is used only to verify you are a real person.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}