# AfriFlow

**Trade Trust Infrastructure for African B2B Commerce**

AfriFlow solves the single root cause of failed intra-African trade: two businesses that have never met cannot safely transact because no neutral trust mechanism exists between them.

The platform provides three things in one system:
1. **Verified Business Profiles** — shareable trust identities
2. **Smart Escrow** — funds held safely until delivery conditions are met
3. **AI-Powered Reputation Engine** — behavioral trust scores that compound with every trade, plus AI-driven dispute resolution

---

## Architecture

```
/afriflow
  /backend          Flask REST API + AI + Trust Score Engine
  /frontend         React + TailwindCSS
```

**Tech Stack**
- Backend: Python, Flask, SQLite, Groq (llama-3.3-70b-versatile)
- Frontend: React, TailwindCSS, Axios
- Payments: Interswitch IPG (simulated escrow)
- AI: Groq Fast Inference API + rule-based fallback

---

## Setup

### Prerequisites
- Python 3.10+
- Node.js 18+
- A Groq API key (free at console.groq.com)

---

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# Run the server
python app.py
```

Backend runs on: http://localhost:5000

On first run, the database is created and demo accounts are seeded automatically.

---

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

Frontend runs on: http://localhost:3000

---

### Groq API Key

1. Go to https://console.groq.com
2. Create a free account
3. Generate an API key
4. Add to `backend/.env`:

```
GROQ_API_KEY=gsk_your_key_here
```

The system includes a full rule-based fallback that works without an API key. The demo is fully runnable offline.

---

## Demo Accounts

| Account | Email | Password | Role |
|---------|-------|----------|------|
| Amaka Clothing Co. | amaka@amakaclothing.ng | demo1234 | Buyer (Lagos) |
| Kwame Asante Textiles | kwame@kwametextiles.gh | demo1234 | Supplier (Accra) |

---

## Running the Demo Scenario (5 minutes)

This is the Amaka → Kwame fabric trade. Run the full loop with two browser windows.

**Window 1: Login as Amaka (Buyer)**
**Window 2: Login as Kwame (Supplier)**

### Step 1 — Amaka views Kwame's profile
- Amaka: Navigate to My Trades → click "View Kwame" quick action
- See Trust Score 87/100, 31 completed trades, delivery history

### Step 2 — Amaka creates a trade
- Amaka: Click "New Trade"
- Select Kwame as supplier
- Fill: "Ankara fabric — mixed colours", "500 metres", ₦280,000, 14 days
- Select release condition: "Timed Auto-Release"
- Submit

### Step 3 — Kwame accepts
- Kwame: Check My Trades — incoming trade appears
- Click trade → "Accept Trade"

### Step 4 — Amaka deposits escrow
- Amaka: Trade dashboard shows "Accepted" status
- Click "Deposit ₦280,000 to Escrow"
- Select payment method → confirm
- Reference number generated — Interswitch params displayed

### Step 5 — Kwame ships
- Kwame: Trade shows "Funded" — funds secured notification
- Click "Upload Shipment Proof"
- Enter tracking number, submit

### Step 6A — Clean resolution
- Amaka: Click "Confirm Delivery"
- Funds release → both Trust Scores update
- View Score Update screen

### Step 6B — Dispute resolution (alternative path)
- Amaka: Click "Raise a Dispute" instead
- Select reason: "Goods arrived damaged"
- Describe issue, submit
- Kwame: Submit counter-evidence from his browser
- Either party: Click "Run AI Analysis"
- Watch AI animate through 6 analysis steps
- See confidence score, finding, recommendation, resolution type

---

## Interswitch Parameters

| Parameter | Value |
|-----------|-------|
| Merchant Code | MX153376 |
| Pay Item ID (Team 1) | 5558761 |

All 10 hackathon team Pay Item IDs are configured in `backend/models.py`.

---

## AI System

### Trust Score Engine (`backend/trust_score.py`)
Rule-based behavioral scoring with ML injection point. Monitors:
- Payment timing reliability
- Delivery consistency
- Dispute frequency and outcomes
- Corridor experience
- Trade volume and frequency

To replace with ML model: swap `compute_signals()` and `score_from_signals()` with a trained XGBoost/LightGBM model. All persistence logic is unchanged.

### Dispute Resolution AI (`backend/dispute_ai.py`)
Sends structured dispute data to Groq (llama-3.3-70b-versatile).
Returns: confidence score (0–100), factual finding, recommendation, resolution type.
Auto-resolves at ≥85% confidence. Escalates to human arbitrator below threshold.
Full rule-based fallback operates without API key.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/auth/login | Login |
| GET | /api/profile/:id | Get trust profile |
| POST | /api/trades | Create trade |
| POST | /api/trades/:id/accept | Accept trade |
| POST | /api/trades/:id/deposit | Deposit to escrow |
| POST | /api/trades/:id/shipment | Upload shipment proof |
| POST | /api/trades/:id/confirm | Confirm delivery |
| POST | /api/trades/:id/dispute | Raise dispute |
| POST | /api/disputes/:id/evidence | Submit evidence |
| POST | /api/disputes/:id/review | Run AI review |
| GET | /api/trust/:id | Get trust score |
| GET | /api/health | Health check |

---

## What This POC Proves

1. Two strangers can complete a trade safely through escrow with full status tracking
2. A behavioral trust score is visible, meaningful, and updates after every trade
3. AI can read dispute evidence and produce a reasoned, confidence-rated finding in under 60 seconds