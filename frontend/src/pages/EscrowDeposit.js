// FILE: frontend/src/pages/EscrowDeposit.js

import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import api from '../api/client';

export default function EscrowDeposit() {
  const { tradeId } = useParams();
  const navigate = useNavigate();
  const [data, setData] = useState(null);
  const [payMethod, setPayMethod] = useState('bank_transfer');
  const [loading, setLoading] = useState(false);
  const [deposited, setDeposited] = useState(false);
  const [reference, setReference] = useState('');

  useEffect(() => {
    api.get(`/trades/${tradeId}`).then(res => setData(res.data));
  }, [tradeId]);

  const handleDeposit = async () => {
  setLoading(true);
  try {
    const res = await api.post(`/trades/${tradeId}/deposit`, { payment_method: payMethod });
    const { gateway_url, payload } = res.data;

    const form = document.createElement('form');
    form.method = 'POST';
    form.action = gateway_url;

    Object.entries(payload).forEach(([key, value]) => {
      const input = document.createElement('input');
      input.type = 'hidden';
      input.name = key;
      input.value = value;
      form.appendChild(input);
    });

    document.body.appendChild(form);
    form.submit();

  } catch (e) {
    alert(e.response?.data?.error || 'Deposit failed.');
    setLoading(false);
  }
};

// Handle return from Interswitch
useEffect(() => {
  const params = new URLSearchParams(window.location.search);
  const paymentStatus = params.get('payment');
  if (paymentStatus === 'success') {
    setDeposited(true);
    setReference(params.get('txnref') || '');
  } else if (paymentStatus === 'failed') {
    alert('Payment failed. Please try again.');
  }
}, []);

  if (!data) return <div className="flex justify-center py-20"><div className="flex gap-1.5"><div className="w-2 h-2 bg-brand-500 rounded-full dot-bounce" /><div className="w-2 h-2 bg-brand-500 rounded-full dot-bounce" /><div className="w-2 h-2 bg-brand-500 rounded-full dot-bounce" /></div></div>;

  const { trade, escrow, supplier } = data;

  if (deposited) {
    return (
      <div className="max-w-lg mx-auto px-6 py-8 fade-in">
        <div className="card text-center space-y-5">
          <div className="w-16 h-16 rounded-full bg-green-500/20 flex items-center justify-center mx-auto">
            <span className="text-3xl">🔐</span>
          </div>
          <div>
            <h2 className="text-2xl font-bold text-white">Funds Secured</h2>
            <p className="text-gray-500 mt-1">
              {trade.currency} {trade.amount?.toLocaleString()} is safely held in escrow.
            </p>
          </div>
          <div className="bg-surface rounded-xl p-4 text-left space-y-2">
            <p className="text-xs text-gray-500">Transaction Reference</p>
            <p className="font-mono text-brand-400 font-medium text-sm">{reference}</p>
          </div>
          <div className="bg-surface rounded-xl p-4 text-left">
            <p className="text-sm text-gray-400">
              <span className="text-white font-medium">{supplier?.business_name}</span> has been notified.
              The funds are protected and will only release when delivery conditions are met.
            </p>
          </div>
          <div className="bg-brand-900/20 border border-brand-800/30 rounded-xl p-3 text-xs text-brand-400">
            Interswitch Reference · Merchant: {escrow?.merchant_code} · Pay Item: {escrow?.pay_item_id}
          </div>
          <button onClick={() => navigate(`/trade/${tradeId}`)} className="btn-primary w-full text-center">
            Back to Trade Dashboard →
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-lg mx-auto px-6 py-8 fade-in space-y-5">
      <div>
        <h1 className="text-2xl font-bold text-white">Secure Your Trade</h1>
        <p className="text-gray-500 text-sm mt-1">Deposit funds into protected escrow to start the trade.</p>
      </div>

      {/* Trade Summary */}
      <div className="card space-y-3">
        <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wide">Trade Summary</h2>
        <Row label="Goods" value={trade.description} />
        <Row label="Quantity" value={trade.quantity} />
        <Row label="Supplier" value={supplier?.business_name} />
        <Row label="Delivery" value={`${trade.delivery_days} days`} />
        <Row label="Release" value={trade.release_condition?.replace(/_/g, ' ')} />
        <div className="border-t border-border pt-3">
          <div className="flex justify-between items-center">
            <span className="text-gray-400">Total to Escrow</span>
            <span className="text-2xl font-bold text-brand-400">
              {trade.currency} {trade.amount?.toLocaleString()}
            </span>
          </div>
        </div>
      </div>

      {/* Interswitch params */}
      <div className="card bg-brand-900/10 border-brand-800/30 space-y-3">
        <h2 className="text-sm font-semibold text-brand-400 uppercase tracking-wide flex items-center gap-2">
          <span>⚡</span> Interswitch Escrow Parameters
        </h2>
        <div className="grid grid-cols-2 gap-3 text-sm">
          <div>
            <p className="text-xs text-gray-500 mb-1">Merchant Code</p>
            <p className="font-mono text-white font-medium">{escrow?.merchant_code}</p>
          </div>
          <div>
            <p className="text-xs text-gray-500 mb-1">Pay Item ID</p>
            <p className="font-mono text-white font-medium">{escrow?.pay_item_id}</p>
          </div>
        </div>
        <p className="text-xs text-gray-600">
          In production, this triggers a real Interswitch IPG payment flow. Funds held until delivery confirmed.
        </p>
      </div>

      {/* Payment method */}
      <div className="card space-y-3">
        <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wide">Pay Via</h2>
        {[
          { value: 'bank_transfer', label: 'Bank Transfer', icon: '🏦' },
          { value: 'mobile_money', label: 'Mobile Money', icon: '📱' },
          { value: 'card', label: 'Debit/Credit Card', icon: '💳' },
        ].map(m => (
          <label key={m.value}
            className={`flex items-center gap-3 p-3 rounded-xl border cursor-pointer transition-all
              ${payMethod === m.value ? 'border-brand-500 bg-brand-900/20' : 'border-border hover:border-gray-600'}`}>
            <input type="radio" name="method" value={m.value}
              checked={payMethod === m.value}
              onChange={e => setPayMethod(e.target.value)}
            />
            <span>{m.icon}</span>
            <span className="text-sm text-white">{m.label}</span>
          </label>
        ))}
      </div>

      <button onClick={handleDeposit} disabled={loading} className="btn-primary w-full text-center text-lg py-4">
        {loading ? (
          <span className="flex items-center justify-center gap-2">
            <div className="w-2 h-2 bg-white rounded-full dot-bounce" />
            <div className="w-2 h-2 bg-white rounded-full dot-bounce" />
            <div className="w-2 h-2 bg-white rounded-full dot-bounce" />
          </span>
        ) : (
          `Deposit ${trade.currency} ${trade.amount?.toLocaleString()} to Escrow →`
        )}
      </button>

      <p className="text-center text-xs text-gray-600">
        Your funds are protected. They will only release when delivery is confirmed or a dispute is resolved.
      </p>
    </div>
  );
}

function Row({ label, value }) {
  return (
    <div className="flex justify-between text-sm">
      <span className="text-gray-500">{label}</span>
      <span className="text-white capitalize">{value}</span>
    </div>
  );
}