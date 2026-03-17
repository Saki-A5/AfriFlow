# FILE: backend/app.py

from flask import Flask, request, jsonify, send_from_directory, redirect
from flask_cors import CORS
from models import db, User, Trade, TrustScore, EscrowAccount, Dispute, Evidence, TradeHistory
from trust_score import calculate_and_update_trust_score, get_trust_profile
from dispute_ai import analyze_dispute_with_ai
from id_verification import verify_id_document, analyze_general_image
import os
import json
import jwt
import datetime
from functools import wraps
from werkzeug.utils import secure_filename
import hashlib
import requests

INTERSWITCH_PAY_URL = "https://sandbox.interswitchng.com/collections/w/pay"
VERIFY_URL = "https://sandbox.interswitchng.com/collections/api/v1/gettransaction.json"
MAC_KEY = "secret"

app = Flask(__name__)
# CORS must include scheme (https://) to match browser Origin exactly.
CORS(app, origins=[
    "http://localhost:3000",
    "https://afri-flow-ytlu.vercel.app",
])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'afriflow.db')

app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_PATH}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'afriflow-dev-secret-2024')
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

db.init_app(app)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not token:
            return jsonify({'error': 'Token missing'}), 401
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            current_user = User.query.get(data['user_id'])
            if not current_user:
                return jsonify({'error': 'User not found'}), 401
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token expired'}), 401
        except Exception:
            return jsonify({'error': 'Invalid token'}), 401
        return f(current_user, *args, **kwargs)
    return decorated


# ─────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    if not data or not data.get('email') or not data.get('password'):
        return jsonify({'error': 'Email and password required'}), 400
    user = User.query.filter_by(email=data['email']).first()
    if user and user.check_password(data['password']):
        token = jwt.encode({
            'user_id': user.id,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
        }, app.config['SECRET_KEY'], algorithm='HS256')
        return jsonify({'token': token, 'user': user.to_dict()})
    return jsonify({'error': 'Invalid credentials'}), 401


@app.route('/api/auth/register', methods=['POST'])
def register():
    """
    Multi-part registration endpoint.
    Accepts multipart/form-data so the ID document can be uploaded in one request.

    Required fields: name, email, password, business_name, phone, location, country
    Optional fields: business_type, registration_number, products_traded,
                     city, whatsapp, website, id_document_type, reg_document
    Files: id_document (required for verification), reg_document (optional)
    """
    # Support both JSON (step-by-step) and multipart (with documents)
    if request.content_type and 'multipart/form-data' in request.content_type:
        data = request.form
    else:
        data = request.get_json() or {}

    # ── Validate required fields ──────────────
    required = ['name', 'email', 'password', 'business_name', 'phone', 'location', 'country']
    missing  = [f for f in required if not data.get(f, '').strip()]
    if missing:
        return jsonify({'error': f'Missing required fields: {", ".join(missing)}'}), 400

    if len(data.get('password', '')) < 8:
        return jsonify({'error': 'Password must be at least 8 characters'}), 400

    if User.query.filter_by(email=data['email'].lower().strip()).first():
        return jsonify({'error': 'An account with this email already exists'}), 409

    # ── Handle document uploads ───────────────
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    id_doc_path  = None
    reg_doc_path = None

    if 'id_document' in request.files:
        f = request.files['id_document']
        if f and f.filename and allowed_file(f.filename):
            fname      = secure_filename(f'id_{data["email"].replace("@","_")}_{f.filename}')
            id_doc_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
            f.save(id_doc_path)

    if 'reg_document' in request.files:
        f = request.files['reg_document']
        if f and f.filename and allowed_file(f.filename):
            fname       = secure_filename(f'reg_{data["email"].replace("@","_")}_{f.filename}')
            reg_doc_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
            f.save(reg_doc_path)

    # ── Create user ───────────────────────────
    user = User(
        name                = data['name'].strip(),
        email               = data['email'].lower().strip(),
        business_name       = data['business_name'].strip(),
        business_type       = data.get('business_type', 'General Trade').strip(),
        registration_number = data.get('registration_number', '').strip() or None,
        products_traded     = data.get('products_traded', '').strip() or None,
        phone               = data['phone'].strip(),
        whatsapp            = data.get('whatsapp', '').strip() or None,
        location            = data['location'].strip(),
        city                = data.get('city', '').strip() or None,
        country             = data['country'].strip(),
        website             = data.get('website', '').strip() or None,
        id_document_path    = id_doc_path,
        id_document_type    = data.get('id_document_type', '').strip() or None,
        reg_document_path   = reg_doc_path,
        verification_status = 'pending',
        profile_complete    = True,
        avatar_initials     = data['business_name'][:2].upper(),
    )
    user.set_password(data['password'])

    # ── AI ID Verification ────────────────────
    id_result = None
    if id_doc_path:
        print(f"[Register] Running AI ID verification for {data['name']}...")
        id_result = verify_id_document(
            file_path     = id_doc_path,
            declared_type = data.get('id_document_type', 'national_id'),
            owner_name    = data['name'].strip(),
        )
        user.id_verification_result     = json.dumps(id_result)
        user.id_verification_confidence = id_result.get('confidence')
        user.id_verification_flags      = ','.join(id_result.get('flags', []))
        user.id_name_extracted          = id_result.get('name_on_document')

        if id_result.get('passed'):
            user.verification_status = 'verified'
            print(f"[Register] ID verified — confidence {id_result.get('confidence')}%")
        else:
            user.verification_status = 'pending'
            user.verification_notes  = id_result.get('rejection_reason', '')
            print(f"[Register] ID not verified — {id_result.get('rejection_reason')}")
    else:
        user.verification_status = 'pending'

    db.session.add(user)
    db.session.flush()   # get user.id before commit

    # ── Generate Trade ID ─────────────────────
    country_code  = _country_code(user.country)
    user.trade_id = f'AFR-{country_code}-{str(user.id).zfill(5)}'

    # ── Bootstrap Trust Score ─────────────────
    trust = TrustScore(
        user_id          = user.id,
        overall_score    = 50.0,
        previous_score   = 50.0,
        score_trajectory = 'insufficient_data',
        score_source     = 'initial',
        score_reasoning  = 'New business — no trade history yet. Score will update after first completed trade.',
    )
    db.session.add(trust)
    db.session.commit()

    token = jwt.encode({
        'user_id': user.id,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
    }, app.config['SECRET_KEY'], algorithm='HS256')

    return jsonify({
        'token':             token,
        'user':              user.to_dict(),
        'id_verification':   id_result,
        'message':           'Account created successfully. Your Trade Profile is live.'
    }), 201


def _country_code(country: str) -> str:
    """Maps country name to 2-letter code for Trade ID generation."""
    codes = {
        'nigeria': 'NG', 'ghana': 'GH', 'kenya': 'KE',
        'south africa': 'ZA', 'ethiopia': 'ET', 'tanzania': 'TZ',
        'uganda': 'UG', 'rwanda': 'RW', 'senegal': 'SN',
        'ivory coast': 'CI', "côte d'ivoire": 'CI',
        'cameroon': 'CM', 'egypt': 'EG', 'morocco': 'MA',
        'zimbabwe': 'ZW', 'zambia': 'ZM', 'mozambique': 'MZ',
    }
    return codes.get(country.lower().strip(), 'AF')


@app.route('/api/auth/me', methods=['GET'])
@token_required
def get_me(current_user):
    return jsonify({'user': current_user.to_dict()})


@app.route('/api/trades/<int:trade_id>/deposit', methods=['POST'])
@token_required
def deposit_escrow(current_user, trade_id):
    trade = Trade.query.get_or_404(trade_id)
    if trade.buyer_id != current_user.id:
        return jsonify({'error': 'Not authorized'}), 403
    if trade.status != 'accepted':
        return jsonify({'error': 'Trade must be accepted before deposit'}), 400

    escrow = EscrowAccount.query.filter_by(trade_id=trade_id).first()
    if not escrow:
        return jsonify({'error': 'Escrow account not found'}), 404

    amount_kobo = int(escrow.amount * 100)
    txn_ref = f'AFR-{trade_id}-{int(datetime.datetime.utcnow().timestamp())}'

    # Save reference before redirecting
    escrow.reference = txn_ref
    db.session.commit()

    # Generate SHA-512 hash
    raw = f"{txn_ref}{escrow.pay_item_id}{amount_kobo}{MAC_KEY}"
    hash_val = hashlib.sha512(raw.encode()).hexdigest()

    payload = {
        "merchantcode": escrow.merchant_code,
        "payitemid": escrow.pay_item_id,
        "txnref": txn_ref,
        "amount": str(amount_kobo),
        "currency": "566",
        "site_redirect_url": f"http://localhost:5000/api/trades/{trade_id}/payment-callback",
        "hash": hash_val,
    }

    return jsonify({"gateway_url": INTERSWITCH_PAY_URL, "payload": payload})


@app.route('/api/trades/<int:trade_id>/payment-callback', methods=['GET', 'POST'])
def payment_callback(trade_id):
    txn_ref = request.args.get('txnref') or request.form.get('txnref')

    trade = Trade.query.get_or_404(trade_id)
    escrow = EscrowAccount.query.filter_by(trade_id=trade_id).first()
    amount_kobo = int(escrow.amount * 100)

    # Verify with Interswitch
    raw = f"{txn_ref}{escrow.pay_item_id}{amount_kobo}{MAC_KEY}"
    hash_val = hashlib.sha512(raw.encode()).hexdigest()
 
    response = requests.get(VERIFY_URL, headers={"Hash": hash_val}, params={
        "merchantcode": escrow.merchant_code,
        "transactionreference": txn_ref,
        "amount": str(amount_kobo),
    })
    result = response.json()

    if result.get("ResponseCode") == "00":
        escrow.status = 'funded'
        trade.status = 'funded'
        trade.updated_at = datetime.datetime.utcnow()
        log_trade_history(trade.buyer_id, trade.id, 'escrow_funded',
                          f'₦{escrow.amount:,.0f} deposited to escrow. Ref: {txn_ref}')
        log_trade_history(trade.supplier_id, trade.id, 'escrow_funded_notify',
                          f'Funds secured in escrow. Ship goods within {trade.delivery_days} days.')
        db.session.commit()
        return redirect(f"http://localhost:3000/trade/{trade_id}/deposit?payment=success&txnref={txn_ref}")
    else:
        return redirect(f"http://localhost:3000/trade/{trade_id}/deposit?payment=failed")



# ─────────────────────────────────────────
# IMAGE VERIFICATION (public endpoints)
# Called from frontend before/after upload
# for instant AI feedback.
# ─────────────────────────────────────────

@app.route('/api/verify/id', methods=['POST'])
def verify_id_preflight():
    """
    Public endpoint — pre-flight ID verification during registration.
    Called immediately when user selects their ID file so they get
    instant feedback before submitting the full registration form.

    Accepts: multipart/form-data with:
      - file         : image file
      - document_type: passport | national_id | drivers_license | voters_card
      - owner_name   : full name of the document holder
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    f = request.files['file']
    if not f or not f.filename or not allowed_file(f.filename):
        return jsonify({'error': 'Invalid file. Upload a JPEG, PNG, or GIF photo of your ID.'}), 400

    declared_type = request.form.get('document_type', 'national_id')
    owner_name    = request.form.get('owner_name', 'Unknown')

    # Save temporarily
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    tmp_filename = secure_filename(f'preflight_id_{f.filename}')
    tmp_path     = os.path.join(app.config['UPLOAD_FOLDER'], tmp_filename)
    f.save(tmp_path)

    result = verify_id_document(
        file_path     = tmp_path,
        declared_type = declared_type,
        owner_name    = owner_name,
    )

    # Clean up temp file — the real one is saved during full registration
    try:
        os.remove(tmp_path)
    except Exception:
        pass

    return jsonify({
        'verification': result,
        'model':        'llama-3.2-11b-vision-preview',
    })


@app.route('/api/verify/image', methods=['POST'])
def analyze_image():
    """
    Public endpoint — general image analysis.
    Used for:
      - Dispute evidence preview (before submitting)
      - Shipment photo verification
      - Any uploaded image that needs AI analysis

    Accepts: multipart/form-data with:
      - file   : image file
      - context: what this image relates to
      - purpose: what this image is supposed to show
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    f = request.files['file']
    if not f or not f.filename or not allowed_file(f.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    context = request.form.get('context', 'a business trade')
    purpose = request.form.get('purpose', 'evidence submission')

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    tmp_filename = secure_filename(f'analysis_{f.filename}')
    tmp_path     = os.path.join(app.config['UPLOAD_FOLDER'], tmp_filename)
    f.save(tmp_path)

    result = analyze_general_image(
        file_path = tmp_path,
        context   = context,
        purpose   = purpose,
    )

    try:
        os.remove(tmp_path)
    except Exception:
        pass

    return jsonify({
        'analysis': result,
        'model':    'llama-3.2-11b-vision-preview',
    })


# ─────────────────────────────────────────
# PROFILES
# ─────────────────────────────────────────

@app.route('/api/profile/<int:user_id>', methods=['GET'])
def get_profile(user_id):
    user = User.query.get_or_404(user_id)
    trust = TrustScore.query.filter_by(user_id=user_id).first()
    history = (TradeHistory.query
               .filter_by(user_id=user_id)
               .order_by(TradeHistory.created_at.desc())
               .limit(20).all())
    return jsonify({
        'user': user.to_dict(),
        'trust_score': trust.to_dict() if trust else get_default_trust(),
        'trade_history': [h.to_dict() for h in history]
    })


@app.route('/api/profile/all', methods=['GET'])
@token_required
def get_all_users(current_user):
    users = User.query.filter(User.id != current_user.id).all()
    result = []
    for u in users:
        trust = TrustScore.query.filter_by(user_id=u.id).first()
        result.append({
            'user': u.to_dict(),
            'trust_score': trust.to_dict() if trust else get_default_trust()
        })
    return jsonify({'users': result})


def get_default_trust():
    return {
        'overall_score': 50,
        'payment_reliability': 50,
        'delivery_accuracy': 50,
        'dispute_rate': 'None',
        'dispute_rate_value': 0,
        'corridor_experience': 'None',
        'total_trades': 0
    }


# ─────────────────────────────────────────
# TRADES
# ─────────────────────────────────────────

@app.route('/api/trades', methods=['POST'])
@token_required
def create_trade(current_user):
    data = request.get_json()
    required = ['supplier_id', 'description', 'quantity', 'amount', 'delivery_days', 'release_condition']
    for field in required:
        if field not in data:
            return jsonify({'error': f'{field} is required'}), 400

    supplier = User.query.get(data['supplier_id'])
    if not supplier:
        return jsonify({'error': 'Supplier not found'}), 404
    if supplier.id == current_user.id:
        return jsonify({'error': 'Cannot trade with yourself'}), 400

    trade = Trade(
        buyer_id=current_user.id,
        supplier_id=data['supplier_id'],
        description=data['description'],
        quantity=data['quantity'],
        amount=float(data['amount']),
        currency=data.get('currency', 'NGN'),
        delivery_days=int(data['delivery_days']),
        release_condition=data['release_condition'],
        status='pending_acceptance'
    )
    db.session.add(trade)
    db.session.flush()

    escrow = EscrowAccount(
        trade_id=trade.id,
        amount=float(data['amount']),
        currency=data.get('currency', 'NGN'),
        status='awaiting_deposit',
        merchant_code='MX6072',      
        pay_item_id='9405967'        
    )
    db.session.add(escrow)

    log_trade_history(current_user.id, trade.id, 'trade_created',
                      f'Trade created for {data["description"]}')
    log_trade_history(data['supplier_id'], trade.id, 'trade_received',
                      f'New trade request from {current_user.business_name}')

    db.session.commit()
    return jsonify({'trade': trade.to_dict(), 'escrow': escrow.to_dict()}), 201


@app.route('/api/trades/<int:trade_id>/accept', methods=['POST'])
@token_required
def accept_trade(current_user, trade_id):
    trade = Trade.query.get_or_404(trade_id)
    if trade.supplier_id != current_user.id:
        return jsonify({'error': 'Not authorized'}), 403
    if trade.status != 'pending_acceptance':
        return jsonify({'error': 'Trade cannot be accepted in current state'}), 400

    trade.status = 'accepted'
    trade.updated_at = datetime.datetime.utcnow()

    log_trade_history(current_user.id, trade.id, 'trade_accepted',
                      'Trade accepted. Waiting for buyer to deposit escrow.')
    log_trade_history(trade.buyer_id, trade.id, 'trade_accepted_notify',
                      f'{current_user.business_name} accepted your trade. Please deposit funds.')

    db.session.commit()
    return jsonify({'trade': trade.to_dict()})

@app.route('/api/trades/<int:trade_id>/shipment', methods=['POST'])
@token_required
def upload_shipment(current_user, trade_id):
    trade = Trade.query.get_or_404(trade_id)
    if trade.supplier_id != current_user.id:
        return jsonify({'error': 'Not authorized'}), 403
    if trade.status != 'funded':
        return jsonify({'error': 'Funds must be in escrow before shipping'}), 400

    data = request.get_json() or {}
    tracking_number = data.get('tracking_number', '')
    notes = data.get('notes', '')

    file_path = None
    if 'file' in request.files:
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(f'shipment_{trade_id}_{file.filename}')
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

    trade.status = 'in_transit'
    trade.tracking_number = tracking_number
    trade.updated_at = datetime.datetime.utcnow()

    log_trade_history(current_user.id, trade.id, 'goods_shipped',
                      f'Goods shipped. Tracking: {tracking_number}')
    log_trade_history(trade.buyer_id, trade.id, 'goods_shipped_notify',
                      f'Goods shipped by {current_user.business_name}. Tracking: {tracking_number}')

    db.session.commit()
    return jsonify({'trade': trade.to_dict()})


@app.route('/api/trades/<int:trade_id>/confirm', methods=['POST'])
@token_required
def confirm_delivery(current_user, trade_id):
    trade = Trade.query.get_or_404(trade_id)
    if trade.buyer_id != current_user.id:
        return jsonify({'error': 'Not authorized'}), 403
    if trade.status not in ['in_transit', 'delivered']:
        return jsonify({'error': 'Goods must be in transit to confirm delivery'}), 400

    escrow = EscrowAccount.query.filter_by(trade_id=trade_id).first()
    escrow.status = 'released'

    trade.status = 'settled'
    trade.updated_at = datetime.datetime.utcnow()

    log_trade_history(current_user.id, trade.id, 'delivery_confirmed',
                      'Delivery confirmed. Funds released to supplier.')
    log_trade_history(trade.supplier_id, trade.id, 'funds_released',
                      f'₦{escrow.amount:,.0f} released to your account.')

    buyer_score = calculate_and_update_trust_score(trade.buyer_id)
    supplier_score = calculate_and_update_trust_score(trade.supplier_id)

    db.session.commit()
    return jsonify({
        'trade': trade.to_dict(),
        'buyer_score': buyer_score,
        'supplier_score': supplier_score
    })


@app.route('/api/trades/<int:trade_id>', methods=['GET'])
@token_required
def get_trade(current_user, trade_id):
    trade = Trade.query.get_or_404(trade_id)
    if trade.buyer_id != current_user.id and trade.supplier_id != current_user.id:
        return jsonify({'error': 'Not authorized'}), 403

    escrow = EscrowAccount.query.filter_by(trade_id=trade_id).first()
    dispute = Dispute.query.filter_by(trade_id=trade_id).first()
    buyer = User.query.get(trade.buyer_id)
    supplier = User.query.get(trade.supplier_id)
    buyer_trust = TrustScore.query.filter_by(user_id=trade.buyer_id).first()
    supplier_trust = TrustScore.query.filter_by(user_id=trade.supplier_id).first()

    return jsonify({
        'trade': trade.to_dict(),
        'escrow': escrow.to_dict() if escrow else None,
        'dispute': dispute.to_dict() if dispute else None,
        'buyer': buyer.to_dict(),
        'supplier': supplier.to_dict(),
        'buyer_trust': buyer_trust.to_dict() if buyer_trust else get_default_trust(),
        'supplier_trust': supplier_trust.to_dict() if supplier_trust else get_default_trust()
    })


@app.route('/api/trades/my', methods=['GET'])
@token_required
def get_my_trades(current_user):
    as_buyer = Trade.query.filter_by(buyer_id=current_user.id).all()
    as_supplier = Trade.query.filter_by(supplier_id=current_user.id).all()
    return jsonify({
        'as_buyer': [t.to_dict() for t in as_buyer],
        'as_supplier': [t.to_dict() for t in as_supplier]
    })


# ─────────────────────────────────────────
# DISPUTES
# ─────────────────────────────────────────

@app.route('/api/trades/<int:trade_id>/dispute', methods=['POST'])
@token_required
def raise_dispute(current_user, trade_id):
    trade = Trade.query.get_or_404(trade_id)
    if trade.buyer_id != current_user.id and trade.supplier_id != current_user.id:
        return jsonify({'error': 'Not authorized'}), 403
    if trade.status not in ['in_transit', 'delivered', 'funded']:
        return jsonify({'error': 'Cannot raise dispute at this trade stage'}), 400

    existing = Dispute.query.filter_by(trade_id=trade_id).first()
    if existing:
        return jsonify({'error': 'Dispute already exists for this trade'}), 400

    data = request.get_json()
    dispute = Dispute(
        trade_id=trade_id,
        raised_by=current_user.id,
        reason=data.get('reason', ''),
        description=data.get('description', ''),
        status='pending_evidence'
    )
    db.session.add(dispute)
    trade.status = 'disputed'
    trade.updated_at = datetime.datetime.utcnow()

    other_party = trade.supplier_id if current_user.id == trade.buyer_id else trade.buyer_id
    log_trade_history(current_user.id, trade.id, 'dispute_raised', f'Dispute raised: {data.get("reason")}')
    log_trade_history(other_party, trade.id, 'dispute_notify', f'A dispute has been raised on this trade.')

    db.session.commit()
    return jsonify({'dispute': dispute.to_dict()}), 201


@app.route('/api/disputes/<int:dispute_id>/evidence', methods=['POST'])
@token_required
def submit_evidence(current_user, dispute_id):
    dispute = Dispute.query.get_or_404(dispute_id)
    trade = Trade.query.get(dispute.trade_id)
    if trade.buyer_id != current_user.id and trade.supplier_id != current_user.id:
        return jsonify({'error': 'Not authorized'}), 403

    file_path = None

    # ── Multipart form (image upload) ──────────────────────
    if request.content_type and 'multipart/form-data' in request.content_type:
        content      = request.form.get('content', '')
        evidence_type = request.form.get('evidence_type', 'image')

        if 'file' in request.files:
            file = request.files['file']
            if file and file.filename and allowed_file(file.filename):
                filename  = secure_filename(f'evidence_{dispute_id}_{current_user.id}_{file.filename}')
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                file.save(file_path)
                print(f"[Upload] Image saved: {file_path}")
            else:
                return jsonify({'error': 'Invalid or unsupported file type'}), 400
    # ── JSON body (text evidence) ───────────────────────────
    else:
        data          = request.get_json() or {}
        content       = data.get('content', '')
        evidence_type = data.get('evidence_type', 'text')

    if not content and not file_path:
        return jsonify({'error': 'Evidence content or file is required'}), 400

    evidence = Evidence(
        dispute_id    = dispute_id,
        submitted_by  = current_user.id,
        evidence_type = evidence_type,
        content       = content,
        file_path     = file_path
    )
    db.session.add(evidence)

    # Mark ready for review once at least one item submitted
    if dispute.status == 'pending_evidence':
        dispute.status = 'ready_for_review'

    log_trade_history(current_user.id, dispute.trade_id, 'evidence_submitted',
                      f'Evidence submitted for dispute ({evidence_type}).')

    db.session.commit()
    return jsonify({'evidence': evidence.to_dict(), 'dispute': dispute.to_dict()}), 201


@app.route('/api/disputes/<int:dispute_id>/review', methods=['POST'])
@token_required
def run_ai_review(current_user, dispute_id):
    dispute = Dispute.query.get_or_404(dispute_id)
    trade = Trade.query.get(dispute.trade_id)
    if trade.buyer_id != current_user.id and trade.supplier_id != current_user.id:
        return jsonify({'error': 'Not authorized'}), 403

    buyer = User.query.get(trade.buyer_id)
    supplier = User.query.get(trade.supplier_id)
    buyer_trust = TrustScore.query.filter_by(user_id=trade.buyer_id).first()
    supplier_trust = TrustScore.query.filter_by(user_id=trade.supplier_id).first()
    all_evidence = Evidence.query.filter_by(dispute_id=dispute_id).all()

    dispute.status = 'ai_reviewing'
    db.session.commit()

    result = analyze_dispute_with_ai(
        trade=trade,
        dispute=dispute,
        buyer=buyer,
        supplier=supplier,
        buyer_trust=buyer_trust,
        supplier_trust=supplier_trust,
        evidence_list=all_evidence
    )

    dispute.ai_confidence      = result['confidence']
    dispute.ai_finding         = result['finding']
    dispute.ai_recommendation  = result['recommendation']
    dispute.ai_resolution_type = result['resolution_type']
    dispute.ai_visual_findings = json.dumps(result.get('visual_findings', []))
    dispute.ai_visual_impact   = result.get('visual_evidence_impact', 'none')
    dispute.ai_vision_model    = result.get('vision_model')
    dispute.ai_text_model      = result.get('text_model')

    if result['confidence'] >= 85:
        dispute.status = 'auto_resolved'
        dispute.resolution = result['recommendation']
        trade.updated_at = datetime.datetime.utcnow()

        escrow = EscrowAccount.query.filter_by(trade_id=trade.id).first()
        if result['resolution_type'] == 'release_to_supplier':
            escrow.status = 'released'
            trade.status = 'settled'
            log_trade_history(trade.supplier_id, trade.id, 'funds_released',
                              'AI resolved dispute. Funds released to supplier.')
            log_trade_history(trade.buyer_id, trade.id, 'dispute_resolved',
                              'AI resolved dispute in favour of supplier.')
        elif result['resolution_type'] == 'refund_to_buyer':
            escrow.status = 'refunded'
            trade.status = 'refunded'
            log_trade_history(trade.buyer_id, trade.id, 'funds_refunded',
                              'AI resolved dispute. Funds refunded to buyer.')
            log_trade_history(trade.supplier_id, trade.id, 'dispute_resolved',
                              'AI resolved dispute in favour of buyer.')

        calculate_and_update_trust_score(trade.buyer_id)
        calculate_and_update_trust_score(trade.supplier_id)
    else:
        dispute.status = 'escalated_to_human'
        trade.status = 'disputed'
        log_trade_history(trade.buyer_id, trade.id, 'escalated',
                          'Dispute escalated to human arbitrator.')
        log_trade_history(trade.supplier_id, trade.id, 'escalated',
                          'Dispute escalated to human arbitrator.')

    db.session.commit()
    return jsonify({
        'dispute': dispute.to_dict(),
        'trade': trade.to_dict(),
        'ai_result': result
    })


@app.route('/api/disputes/<int:dispute_id>', methods=['GET'])
@token_required
def get_dispute(current_user, dispute_id):
    dispute = Dispute.query.get_or_404(dispute_id)
    evidence = Evidence.query.filter_by(dispute_id=dispute_id).all()
    return jsonify({
        'dispute': dispute.to_dict(),
        'evidence': [e.to_dict() for e in evidence]
    })


# ─────────────────────────────────────────
# TRUST SCORES
# ─────────────────────────────────────────

@app.route('/api/trust/<int:user_id>', methods=['GET'])
def get_trust_score(user_id):
    trust = TrustScore.query.filter_by(user_id=user_id).first()
    if not trust:
        return jsonify(get_default_trust())
    return jsonify(trust.to_dict())


@app.route('/api/trust/<int:user_id>/recalculate', methods=['POST'])
@token_required
def recalculate_trust(current_user, user_id):
    score = calculate_and_update_trust_score(user_id)
    return jsonify(score)


# ─────────────────────────────────────────
# UPLOADS
# ─────────────────────────────────────────

@app.route('/api/uploads/<filename>')
def get_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# ─────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'AfriFlow API', 'version': '1.0.0'})


def log_trade_history(user_id, trade_id, event_type, description):
    entry = TradeHistory(
        user_id=user_id,
        trade_id=trade_id,
        event_type=event_type,
        description=description
    )
    db.session.add(entry)


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        print("[AfriFlow] Database tables created.")
        print("[AfriFlow] API running at http://localhost:5000")
    app.run(debug=True, port=5000)