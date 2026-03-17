// FILE: frontend/src/App.js

import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';

import Landing from './pages/Landing';
import Register from './pages/Register';
import TrustProfile from './pages/TrustProfile';
import CreateTrade from './pages/CreateTrade';
import TradeDashboard from './pages/TradeDashboard';
import EscrowDeposit from './pages/EscrowDeposit';
import ShipmentUpload from './pages/ShipmentUpload';
import DeliveryConfirmation from './pages/DeliveryConfirmation';
import DisputeSubmission from './pages/DisputeSubmission';
import AIDisputeReview from './pages/AIDisputeReview';
import TrustScoreUpdate from './pages/TrustScoreUpdate';
import MyTrades from './pages/MyTrades';
import Navbar from './components/Navbar';

// ─────────────────────────────────────────────────────────────
// Guards
// ─────────────────────────────────────────────────────────────

/** Redirects unauthenticated users to / */
function PrivateRoute({ children }) {
  const { user, loading } = useAuth();
  if (loading) return <LoadingDots />;
  return user ? children : <Navigate to="/" replace />;
}

/** Redirects already-logged-in users away from auth pages */
function GuestRoute({ children }) {
  const { user, loading } = useAuth();
  if (loading) return <LoadingDots />;
  return user ? <Navigate to="/my-trades" replace /> : children;
}

function LoadingDots() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-surface">
      <div className="flex gap-2">
        <div className="w-2 h-2 bg-brand-500 rounded-full dot-bounce" />
        <div className="w-2 h-2 bg-brand-500 rounded-full dot-bounce" />
        <div className="w-2 h-2 bg-brand-500 rounded-full dot-bounce" />
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// Layout — Navbar only visible when logged in.
// Auth pages (/ and /register) manage their own nav.
// ─────────────────────────────────────────────────────────────

const AUTH_PATHS = ['/', '/register'];

function AppLayout({ children }) {
  const { user } = useAuth();
  const isAuthPage = AUTH_PATHS.includes(window.location.pathname);

  return (
    <div className="min-h-screen bg-surface">
      {user && !isAuthPage && <Navbar />}
      <main className={user && !isAuthPage ? 'pt-16' : ''}>
        {children}
      </main>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// Routes
// ─────────────────────────────────────────────────────────────

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <AppLayout>
          <Routes>

            {/* ── Public auth pages ─────────────────────── */}
            <Route path="/" element={
              <GuestRoute><Landing /></GuestRoute>
            } />
            <Route path="/register" element={
              <GuestRoute><Register /></GuestRoute>
            } />

            {/* ── Public Trade Profiles ─────────────────────
                Anyone with a profile link can view it.
                The profile page handles its own CTA logic
                (shows "Start a Trade" if logged in,
                "Sign up to trade with X" if not).        */}
            <Route path="/profile/:userId" element={<TrustProfile />} />

            {/* ── Authenticated trade flow ───────────────── */}
            <Route path="/my-trades" element={
              <PrivateRoute><MyTrades /></PrivateRoute>
            } />
            <Route path="/trade/create" element={
              <PrivateRoute><CreateTrade /></PrivateRoute>
            } />
            <Route path="/trade/:tradeId" element={
              <PrivateRoute><TradeDashboard /></PrivateRoute>
            } />
            <Route path="/trade/:tradeId/deposit" element={
              <PrivateRoute><EscrowDeposit /></PrivateRoute>
            } />
            <Route path="/trade/:tradeId/shipment" element={
              <PrivateRoute><ShipmentUpload /></PrivateRoute>
            } />
            <Route path="/trade/:tradeId/confirm" element={
              <PrivateRoute><DeliveryConfirmation /></PrivateRoute>
            } />
            <Route path="/trade/:tradeId/dispute" element={
              <PrivateRoute><DisputeSubmission /></PrivateRoute>
            } />
            <Route path="/dispute/:disputeId/review" element={
              <PrivateRoute><AIDisputeReview /></PrivateRoute>
            } />
            <Route path="/trade/:tradeId/complete" element={
              <PrivateRoute><TrustScoreUpdate /></PrivateRoute>
            } />

            {/* ── Fallback ───────────────────────────────── */}
            <Route path="*" element={<Navigate to="/" replace />} />

          </Routes>
        </AppLayout>
      </BrowserRouter>
    </AuthProvider>
  );
}