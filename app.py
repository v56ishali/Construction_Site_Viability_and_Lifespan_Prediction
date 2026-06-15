# app.py — Flask Backend

import os
from dotenv import load_dotenv # pyre-ignore
load_dotenv()
from flask import Flask, request, jsonify, render_template, send_file # pyre-ignore
from flask_cors import CORS # pyre-ignore
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity # pyre-ignore
from flask_sqlalchemy import SQLAlchemy # pyre-ignore
from flask_limiter import Limiter # pyre-ignore
from flask_limiter.util import get_remote_address # pyre-ignore
from flask_mail import Mail, Message # pyre-ignore
from predictor import predict_location, HIST_DATA_PATH, get_hist_path # pyre-ignore
from io import BytesIO
from datetime import datetime
import json
import os
import pandas as pd # pyre-ignore
from werkzeug.utils import secure_filename # pyre-ignore
from werkzeug.security import generate_password_hash, check_password_hash # pyre-ignore
from reportlab.pdfgen import canvas # pyre-ignore
from reportlab.lib.pagesizes import A4 # pyre-ignore
import requests # pyre-ignore

def get_scenario_filename(result):
    """Maps the analysis result to one of 10 pre-rendered Image Scenarios"""
    raw = result.get('raw_data', {})
    env = raw.get('env', {})
    climate = raw.get('climate', {})
    soil = raw.get('soil', {})
    animal = raw.get('animal', {})
    
    # Parse factors
    flood_risk = str(env.get('flood_risk') or climate.get('flood_risk') or 'Low').upper()
    seismic_risk = str(env.get('earthquake_risk') or 'Low').upper()
    seismic_high = seismic_risk in ["HIGH", "SEVERE", "ZONE V", "ZONE IV"]
    
    bearing = soil.get('bearing_capacity_kNm2', 150)
    try: bearing = float(bearing)
    except: bearing = 150
    soil_weak = bearing < 100
    
    wildlife_high = str(animal.get('protected_area_risk') or 'Low').upper() == "HIGH"
    
    # Map to EXACT 10 scenarios
    # 1. Extreme Hazard (3+ risks)
    risk_count = (flood_risk == "HIGH") + seismic_high + soil_weak + wildlife_high
    if risk_count >= 3:
        return "scenario_extreme_hazard.png"
        
    # Combinations (2 risks)
    if flood_risk == "HIGH" and seismic_high: return "scenario_flood_and_seismic.png"
    if flood_risk == "HIGH" and soil_weak: return "scenario_flood_and_soil.png"
    if seismic_high and soil_weak: return "scenario_seismic_and_soil.png"
    if risk_count == 2: return "scenario_extreme_hazard.png" # Fallback for other 2-combos
    
    # Single Risks
    if flood_risk == "HIGH": return "scenario_flood_heavy.png"
    if seismic_high: return "scenario_seismic_damage.png"
    if soil_weak: return "scenario_soil_weak.png"
    if wildlife_high: return "scenario_wildlife.png"
    
    # 0 High Risks
    # Check for mediums
    flood_med = flood_risk == "MEDIUM"
    seismic_med = seismic_risk in ["MEDIUM", "ZONE III"]
    wildlife_med = str(animal.get('protected_area_risk') or 'Low').upper() == "MEDIUM"
    if flood_med or seismic_med or wildlife_med or (bearing < 150):
        return "scenario_moderate_risk.png"
        
    # Safe
    return "scenario_ideal.png"

app = Flask(__name__)
CORS(app)

default_sqlite_path = os.path.join(os.path.dirname(__file__), 'data', 'app.db')
default_sqlite = f"sqlite:///{default_sqlite_path.replace(os.sep, '/')}"
db_url = os.getenv("DATABASE_URL", default_sqlite)

if db_url.startswith("sqlite:///"):
    # Extract the actual file path from the URI
    # This handles both sqlite:///path and sqlite:////path
    db_file_path = db_url.replace("sqlite:///", "").replace("/", os.sep)
    db_dir = os.path.dirname(db_file_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY", "change-me")
app.config["MAIL_SERVER"] = os.getenv("MAIL_SERVER", "")
app.config["MAIL_PORT"] = int(os.getenv("MAIL_PORT", "587"))
app.config["MAIL_USE_TLS"] = os.getenv("MAIL_USE_TLS", "1").lower() in {"1", "true", "yes"}
app.config["MAIL_USERNAME"] = os.getenv("MAIL_USERNAME", "")
app.config["MAIL_PASSWORD"] = os.getenv("MAIL_PASSWORD", "")
app.config["MAIL_DEFAULT_SENDER"] = os.getenv("MAIL_DEFAULT_SENDER", "")

db = SQLAlchemy(app)
jwt = JWTManager(app)
mail = Mail(app)
limiter = Limiter(get_remote_address, app=app, default_limits=["200 per day", "50 per hour"])

AUTH_REQUIRED = os.getenv("AUTH_REQUIRED", "1").lower() in {"1", "true", "yes"}
AUDIT_DIR = os.path.join(os.path.dirname(__file__), "logs")
AUDIT_FILE = os.path.join(AUDIT_DIR, "audit_log.jsonl")
REVIEW_FILE = os.path.join(AUDIT_DIR, "review_log.jsonl")


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __init__(self, **kwargs):
        super(User, self).__init__(**kwargs)


with app.app_context():
    db.create_all()


def _maybe_jwt_required(fn):
    if AUTH_REQUIRED:
        return jwt_required()(fn)
    return fn

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/login")
def login_page():
    return render_template("login.html")


@app.route("/register")
def register_page():
    return render_template("register.html")

@app.route("/visualization", methods=["GET", "POST"])
def visualization_page():
    if request.method == "POST":
        try:
            # Similar to analyze but directly returns the page with image
            lat           = float(request.form.get("lat", 0))
            lon           = float(request.form.get("lon", 0))
            building_type = request.form.get("building_type", "House")
            floors        = int(request.form.get("floors", 2))
            
            if not (6.5 <= lat <= 37.5 and 67.0 <= lon <= 97.5):
                return render_template("visualization.html", error="Location must be within India!")
                
            land_status = _land_status(lat, lon)
            if land_status == "water":
                return render_template("visualization.html", error="Selected point is water.")
            if land_status == "unknown":
                return render_template(
                    "visualization.html",
                    error="Unable to verify land vs water for this location. Please try again or select another point."
                )
                
            pred_result = predict_location(lat, lon, building_type, floors, sensor_data={})
            
            # Map prediction to one of 10 visual scenarios
            scenario_filename = get_scenario_filename(pred_result)
            image_path = f"/static/scenarios/{scenario_filename}"
            
            # Inject visualization helpers for badges & CSS animation
            raw = pred_result.get('raw_data', {})
            env_data = raw.get('env', {})
            animal_data = raw.get('animal', {})
            soil_data = raw.get('soil', {})
            pred_result['flood_risk'] = str(env_data.get('flood_risk', 'LOW')).upper()
            pred_result['seismic_risk'] = str(env_data.get('earthquake_risk', 'LOW')).upper()
            
            bearing = soil_data.get('bearing_capacity_kNm2', 150)
            try: bearing = float(bearing)
            except: bearing = 150
            pred_result['soil_strength'] = 'WEAK' if bearing < 100 else 'STRONG'
            
            pa_risk = str(animal_data.get('protected_area_risk', 'LOW')).upper()
            pred_result['animal_conflict'] = pa_risk
            
            return render_template("visualization.html", result=pred_result, image_path=image_path)
        except Exception as e:
            return render_template("visualization.html", error=str(e))
            
            
    return render_template("visualization.html", result=None, image_path=None)

@app.route("/api/analyze", methods=["POST"])
@limiter.limit("30 per minute")
@_maybe_jwt_required
def analyze():
    try:
        data          = request.get_json()
        lat           = float(data["lat"])
        lon           = float(data["lon"])
        building_type = data.get("building_type", "House")
        floors        = int(data.get("floors", 2))
        sensor_data   = data.get("sensor_data") or {}

        # Validate India bounds
        if not (6.5 <= lat <= 37.5 and 67.0 <= lon <= 97.5):
            return jsonify({"error": "Location must be within India!"}), 400

        # Basic water-body guard using OpenStreetMap reverse geocode
        land_status = _land_status(lat, lon)
        if land_status == "water":
            return jsonify({"error": "Selected point appears to be water. Choose a land location."}), 400
        if land_status == "unknown":
            return jsonify({"error": "Unable to verify land vs water for this location. Please try again."}), 400

        result = predict_location(lat, lon, building_type, floors, sensor_data=sensor_data)
        user_identity = None
        try:
            user_identity = get_jwt_identity()
        except Exception:
            user_identity = None
        _write_audit_log({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "ip": request.remote_addr,
            "user": user_identity,
            "inputs": {
                "lat": lat,
                "lon": lon,
                "building_type": building_type,
                "floors": floors,
                "sensor_data": sensor_data
            },
            "result": result
        })
        return jsonify({"success": True, "result": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/voice-report", methods=["POST"])
@limiter.limit("20 per minute")
@_maybe_jwt_required
def voice_report():
    try:
        data = request.get_json()
        print("DEBUG voice-report received:", data)
        if not data:
            return jsonify({"error": "No data received"}), 400
        
        result = data.get("result", {})
        if not result:
            # Fallback to checking if results were sent directly
            result = data
            
        script = _generate_tamil_script(result)
        print("DEBUG script generated:", str(script)[:100] + "...") # pyre-ignore
        return jsonify({"success": True, "script": script})
    except Exception as e:
        print("ERROR in voice-report:", str(e))
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

def _generate_tamil_script(result):
    def safe_float(val, default=0.0):
        if val is None: return default
        if isinstance(val, (int, float)): return float(val)
        try:
            import re
            # Find the first sequence that looks like a number
            match = re.search(r"[-+]?\d*\.\d+|\d+", str(val).replace('–', '-'))
            return float(match.group()) if match else default
        except:
            return default

    # ── SAFE VALUE EXTRACTION ──
    score = safe_float(result.get('final_feasibility_score') 
               or result.get('feasibility_score') 
               or result.get('score'), 50.0)
    
    # Handle lifespan ranges like '10–23 years'
    lifespan_val = result.get('predicted_lifespan') or result.get('lifespan')
    lifespan_min = safe_float(result.get('lifespan_min'))
    lifespan_max = safe_float(result.get('lifespan_max'))
    
    if not lifespan_min or not lifespan_max:
        import re
        nums = re.findall(r"\d+", str(lifespan_val or "60").replace('–', '-'))
        if len(nums) >= 2:
            lifespan_min = float(nums[0])
            lifespan_max = float(nums[1])
        elif len(nums) == 1:
            val = float(nums[0])
            lifespan_min = max(0.0, val - 15.0)
            lifespan_max = val + 15.0
        else:
            lifespan_min, lifespan_max = 50.0, 75.0
    
    bearing = safe_float(result.get('bearing_capacity_kNm2') 
               or result.get('bearing_capacity'), 100.0)
    
    foundation = str(
        result.get('foundation_recommendation') 
        or result.get('foundation') 
        or 'Isolated Footing')
    
    flood = str(result.get('flood_risk') or 'Low')
    earthquake = str(result.get('earthquake_risk') or 'Low')
    cyclone = str(result.get('cyclone_risk') or 'None')
    elephant = str(result.get('elephant_corridor_risk') or 'Low')
    pa_risk = str(result.get('protected_area_risk') or 'Low')
    
    lat = safe_float(
        result.get('latitude') 
        or result.get('lat')
        or result.get('input_lat')
        or result.get('query_lat'), 0.0)

    lon = safe_float(
        result.get('longitude')
        or result.get('lon') 
        or result.get('input_lon')
        or result.get('query_lon'), 0.0)
    
    location_name = result.get('location_name')
    
    confidence = safe_float(result.get('confidence_percent') or result.get('confidence'), 75.0)
    
    # ── FEASIBILITY TEXT ──
    if score >= 75:
        feasibility_text = (
            f"இந்த இடத்தின் சாத்தியக்கூறு "
            f"மதிப்பெண் {score:.1f} சதவீதம். "
            f"இது நல்ல மதிப்பெண். "
            f"இங்கே கட்டுமானம் மேற்கொள்ளலாம்.")
    elif score >= 50:
        feasibility_text = (
            f"இந்த இடத்தின் சாத்தியக்கூறு "
            f"மதிப்பெண் {score:.1f} சதவீதம். "
            f"இது நடுத்தர மதிப்பெண். "
            f"கவனமாக திட்டமிட்டு கட்டலாம்.")
    else:
        feasibility_text = (
            f"இந்த இடத்தின் சாத்தியக்கூறு "
            f"மதிப்பெண் {score:.1f} சதவீதம். "
            f"இது குறைவான மதிப்பெண். "
            f"இங்கே கட்டுமானம் தவிர்க்க "
            f"பரிந்துரைக்கிறோம்.")
    
    # ── LIFESPAN TEXT ──
    if lifespan_max >= 80:
        lifespan_text = (
            f"இந்த கட்டிடம் சுமார் "
            f"{lifespan_min:.0f} முதல் "
            f"{lifespan_max:.0f} ஆண்டுகள் வரை "
            f"நீடிக்கும். இது மிகவும் நல்ல "
            f"ஆயுட்காலம்.")
    elif lifespan_max >= 50:
        lifespan_text = (
            f"இந்த கட்டிடம் சுமார் "
            f"{lifespan_min:.0f} முதல் "
            f"{lifespan_max:.0f} ஆண்டுகள் வரை "
            f"நீடிக்கும். சராசரி ஆயுட்காலம்.")
    else:
        lifespan_text = (
            f"இந்த கட்டிடம் சுமார் "
            f"{lifespan_min:.0f} முதல் "
            f"{lifespan_max:.0f} ஆண்டுகள் மட்டுமே "
            f"நீடிக்கும். இது மிகவும் குறைவு. "
            f"ஆழமான அடித்தளம் அவசியம்.")
    
    # ── BEARING CAPACITY TEXT ──
    if bearing >= 150:
        soil_text = (
            f"மண்ணின் சுமை தாங்கும் திறன் "
            f"{bearing:.0f} கிலோ நியூட்டன் "
            f"சதுர மீட்டர். இது சிறந்த மண் தரம். "
            f"எந்த வகை கட்டிடமும் கட்டலாம்.")
    elif bearing >= 60:
        soil_text = (
            f"மண்ணின் சுமை தாங்கும் திறன் "
            f"{bearing:.0f} கிலோ நியூட்டன் "
            f"சதுர மீட்டர். சராசரி மண் தரம். "
            f"சரியான அடித்தளத்துடன் கட்டலாம்.")
    else:
        soil_text = (
            f"மண்ணின் சுமை தாங்கும் திறன் "
            f"{bearing:.0f} கிலோ நியூட்டன் "
            f"சதுர மீட்டர். இது பலவீனமான மண். "
            f"ஆழமான பைல் அடித்தளம் தேவை.")
    
    # ── FOUNDATION TEXT ──
    foundation_map = {
        'Pile Foundation': 
            'பைல் அடித்தளம். மண்ணின் கீழே '
            'உள்ள பாறை வரை ஆழமாக போக வேண்டும்.',
        'Pile Foundation (Deep)': 
            'ஆழமான பைல் அடித்தளம். '
            'பாறை அடுக்கு வரை தோண்ட வேண்டும்.',
        'Raft Foundation': 
            'ராஃப்ட் அடித்தளம். '
            'பரந்த தட்டு போன்ற அடித்தளம்.',
        'Isolated Footing with RCC': 
            'RCC தனி அடித்தளம். '
            'கனமான கட்டிடங்களுக்கு ஏற்றது.',
        'Isolated Footing': 
            'தனி அடித்தளம். '
            'சாதாரண வீடுகளுக்கு ஏற்றது.',
        'Simple Strip Footing': 
            'நேர் கோடு அடித்தளம். '
            'நல்ல மண்ணில் எளிய வீடுகளுக்கு.',
    }
    foundation_tamil = foundation_map.get(
        foundation, 
        f'{foundation} வகை அடித்தளம்.')
    
    # ── RISK TEXTS ──
    risk_texts = []
    
    if flood.lower() == 'high':
        risk_texts.append(
            "வெள்ள அபாயம் அதிகமாக உள்ளது. "
            "தரை மட்டத்திலிருந்து குறைந்தது "
            "600 மிமீ உயரத்தில் கட்டவும்.")
    elif flood.lower() == 'medium':
        risk_texts.append(
            "மிதமான வெள்ள அபாயம் உள்ளது. "
            "வடிகால் ஏற்பாடு செய்யவும்.") # Fixed Korean typo "배수" to Tamil "வடிகால்"
    
    if earthquake.lower() == 'high':
        risk_texts.append(
            "நிலநடுக்க அபாயம் அதிகமாக உள்ளது. "
            "IS 1893 தரநிலைப்படி கட்ட வேண்டும்.")
    elif earthquake.lower() == 'medium':
        risk_texts.append(
            "நிலநடுக்க அபாயம் உள்ளது. "
            "பலப்படுத்தப்பட்ட கட்டமைப்பு தேவை.")
    
    if cyclone.lower() not in ['none','low','no']:
        risk_texts.append(
            "புயல் அபாயம் உள்ளது. "
            "கூரை மிகவும் வலுவாக இருக்க வேண்டும்.")
    
    if elephant.lower() == 'high':
        risk_texts.append(
            "எச்சரிக்கை! இந்த இடம் யானை "
            "நடைபாதையில் உள்ளது. "
            "வனத்துறை அனுமதி அவசியம்.")
    
    if pa_risk.lower() == 'high':
        risk_texts.append(
            "இந்த இடம் பாதுகாக்கப்பட்ட "
            "காடுகளுக்கு அருகில் உள்ளது. "
            "சட்டரீதியான அனுமதி பெறவும்.")
    
    if not risk_texts:
        risk_texts.append(
            "பெரிய இயற்கை அபாயங்கள் எதுவும் "
            "கண்டறியப்படவில்லை.")
    
    risks_combined = " ".join(risk_texts)
    
    # ── VERDICT ──
    if score >= 75 and lifespan_max >= 60:
        verdict = (
            "மொத்தத்தில், இந்த இடம் "
            "கட்டுமானத்திற்கு தகுதியானது. "
            "நல்ல திட்டமிடலுடன் கட்டுமானத்தை "
            "தொடங்கலாம். உங்கள் கட்டுமான பணி "
            "வெற்றிகரமாக அமைய வாழ்த்துக்கள்!")
    else:
        verdict = (
            "மொத்தத்தில், இந்த இடத்தில் சில "
            "கவலைகள் உள்ளன. தகுதிவாய்ந்த "
            "சிவில் இன்ஜினியரை அணுகி "
            "மேலும் ஆலோசனை பெறவும்.")
    
    location_text = ""
    if location_name:
        location_text = f"நீங்கள் பகுப்பாய்வு செய்த இடம், {location_name} பகுதியில் அமைந்துள்ளது. "
    elif not (lat == 0 and lon == 0):
        # Force float to satisfy strict linters before using abs()
        f_lat = float(lat)
        f_lon = float(lon)
        location_text = (
            f"நீங்கள் பகுப்பாய்வு செய்த இடம், "
            f"{abs(f_lat):.2f} டிகிரி "
            f"{'வடக்கு' if f_lat >= 0 else 'தெற்கு'} "
            f"மற்றும் {abs(f_lon):.2f} டிகிரி "
            f"{'கிழக்கு' if f_lon >= 0 else 'மேற்கு'} "
            f"ஆயத்தொலைவில் அமைந்துள்ளது. ")

    # ── ASSEMBLE FULL SCRIPT ──
    script = (
        f"வணக்கம்! நான் கட்டுமான தள AI "
        f"உதவியாளர். "
        f"நீங்கள் தேர்வு செய்த இடத்தின் "
        f"முழு அறிக்கையை இப்போது கேட்கலாம். "
        f"{location_text}"
        f"{feasibility_text} "
        f"{lifespan_text} "
        f"{soil_text} "
        f"பரிந்துரைக்கப்படும் அடித்தளம்: "
        f"{foundation_tamil} "
        f"{risks_combined} "
        f"{verdict} "
        f"இது AI அடிப்படையிலான மதிப்பீடு "
        f"மட்டுமே. இறுதி முடிவிற்கு "
        f"அங்கீகரிக்கப்பட்ட சிவில் "
        f"இன்ஜினியரின் ஆலோசனை அவசியம். "
        f"நன்றி!")
    
    return script

@app.route("/api/report", methods=["POST"])
@limiter.limit("10 per minute")
@_maybe_jwt_required
def report():
    try:
        data = request.get_json() or {}
        inputs = data.get("inputs", {})
        result = data.get("result", {})
        review = data.get("review", {})
        if not inputs or not result:
            return jsonify({"error": "Missing inputs or result"}), 400

        pdf_bytes = _build_report_pdf(inputs, result, review)
        return send_file(
            pdf_bytes,
            mimetype="application/pdf",
            as_attachment=True,
            download_name="site_feasibility_report.pdf"
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/review", methods=["POST"])
@limiter.limit("10 per minute")
@_maybe_jwt_required
def review():
    try:
        data = request.get_json() or {}
        inputs = data.get("inputs", {})
        result = data.get("result", {})
        review_data = data.get("review", {})
        if not review_data:
            return jsonify({"error": "Missing review data"}), 400

        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "ip": request.remote_addr,
            "inputs": inputs,
            "result": result,
            "review": review_data
        }
        _write_review_log(payload)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "models": "loaded"})

@app.route("/api/datasets/status", methods=["GET"])
@limiter.limit("20 per minute")
@_maybe_jwt_required
def dataset_status():
    try:
        hist_path = get_hist_path()
        if not os.path.exists(hist_path):
            return jsonify({"exists": False})
        df = pd.read_csv(hist_path)
        return jsonify({
            "exists": True,
            "rows": int(len(df)),
            "columns": list(df.columns),
            "updated": datetime.utcfromtimestamp(os.path.getmtime(hist_path)).isoformat() + "Z"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/datasets/upload", methods=["POST"])
@limiter.limit("5 per minute")
@_maybe_jwt_required
def dataset_upload():
    try:
        if "file" not in request.files:
            return jsonify({"error": "Missing file"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400
        fname = secure_filename(file.filename)
        if not fname.lower().endswith(".csv"):
            return jsonify({"error": "Only CSV files supported"}), 400
        os.makedirs(os.path.dirname(HIST_DATA_PATH), exist_ok=True)
        file.save(HIST_DATA_PATH)
        return jsonify({"success": True, "path": "data/historical_data.csv"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/report/email", methods=["POST"])
@limiter.limit("5 per minute")
@_maybe_jwt_required
def report_email():
    try:
        data = request.get_json() or {}
        to_email = data.get("to_email")
        inputs = data.get("inputs", {})
        result = data.get("result", {})
        review = data.get("review", {})
        if not to_email:
            return jsonify({"error": "Missing to_email"}), 400
        if not inputs or not result:
            return jsonify({"error": "Missing inputs or result"}), 400
        if not app.config.get("MAIL_SERVER"):
            print(f"Mock sending email to {to_email}")
            return jsonify({"success": True, "message": "Email sent successfully (mocked)"})

        pdf_bytes = _build_report_pdf(inputs, result, review)
        msg = Message("AI Construction Site Feasibility Report", recipients=[to_email])
        msg.body = "Please find the attached feasibility report."
        msg.attach("site_feasibility_report.pdf", "application/pdf", pdf_bytes.getvalue())
        mail.send(msg)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/auth/register", methods=["POST"])
@limiter.limit("5 per minute")
def register():
    try:
        data = request.get_json() or {}
        email = (data.get("email") or "").strip().lower()
        password = data.get("password") or ""
        if not email or not password:
            return jsonify({"error": "Missing email or password"}), 400
        if User.query.filter_by(email=email).first():
            return jsonify({"error": "User already exists"}), 400
        
        user = User()
        user.email = email
        user.password_hash = generate_password_hash(password)
        
        db.session.add(user)
        db.session.commit()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/auth/login", methods=["POST"])
@limiter.limit("10 per minute")
def login():
    try:
        data = request.get_json() or {}
        email = (data.get("email") or "").strip().lower()
        password = data.get("password") or ""
        user = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password_hash, password):
            return jsonify({"error": "Invalid credentials"}), 401
        token = create_access_token(identity=email)
        return jsonify({"access_token": token})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def _write_audit_log(payload):
    os.makedirs(AUDIT_DIR, exist_ok=True)
    with open(AUDIT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")

def _write_review_log(payload):
    os.makedirs(AUDIT_DIR, exist_ok=True)
    with open(REVIEW_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")

def _build_report_pdf(inputs, result, review=None):
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    y = height - 60
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "AI Construction Site Feasibility Report")
    y -= 24
    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Generated: {datetime.utcnow().isoformat()}Z")
    y -= 20

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Input Location")
    y -= 16
    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Latitude: {inputs.get('lat')} | Longitude: {inputs.get('lon')}")
    y -= 14
    c.drawString(50, y, f"Construction Type: {inputs.get('building_type')} | Floors: {inputs.get('floors')}")
    y -= 22

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Key Outputs")
    y -= 16
    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Feasibility Score: {result.get('feasibility_score')}%")
    y -= 14
    c.drawString(50, y, f"Risk Level: {result.get('risk_level')}")
    y -= 14
    c.drawString(50, y, f"Lifespan: {result.get('lifespan')} | Confidence: {result.get('confidence')}%")
    y -= 14
    c.drawString(50, y, f"Foundation Recommendation: {result.get('foundation')}")
    y -= 14
    c.drawString(50, y, f"Risk Factor Summary: {result.get('risk_factor_summary')}")
    y -= 22

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Model Scores")
    y -= 16
    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Soil Degradation Risk: {result.get('soil_degradation_risk_score')}%")
    y -= 14
    c.drawString(50, y, f"Climate Stress Frequency: {result.get('climate_stress_frequency_score')}%")
    y -= 14
    c.drawString(50, y, f"Water Exposure Probability: {result.get('water_exposure_probability_score')}%")
    y -= 14
    c.drawString(50, y, f"Biological Damage Probability: {result.get('biological_damage_probability_score')}%")
    y -= 22

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "AHP Comparison")
    y -= 16
    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"AHP Weighted Score: {result.get('ahp_score')}%")
    y -= 14
    c.drawString(50, y, f"AI vs AHP Delta: {result.get('ahp_delta')}%")
    y -= 22

    domain = result.get("domain_scores", {})
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Domain Scores")
    y -= 16
    c.setFont("Helvetica", 10)
    c.drawString(
        50,
        y,
        f"Soil: {domain.get('soil')}%  Climate: {domain.get('climate')}%  Env: {domain.get('environment')}%  Animal: {domain.get('animal')}%"
    )
    y -= 22

    if review:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Engineer Review")
        y -= 16
        c.setFont("Helvetica", 10)
        c.drawString(50, y, f"Reviewer: {review.get('reviewer_name', '-')}")
        y -= 14
        c.drawString(50, y, f"License ID: {review.get('license_id', '-')}")
        y -= 14
        c.drawString(50, y, f"Decision: {review.get('decision', '-')}")
        y -= 14
        c.drawString(50, y, f"Review Date: {review.get('review_date', '-')}")
        y -= 14
        notes = review.get("notes", "-")
        c.drawString(50, y, f"Notes: {notes[:120]}")
        y -= 22

        checklist = review.get("checklist", {})
        if checklist:
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y, "Checklist Status")
            y -= 16
            c.setFont("Helvetica", 9)
            items = [
                ("Soil bearing test", checklist.get("soil_bearing_test")),
                ("Groundwater survey", checklist.get("groundwater_survey")),
                ("Seismic check", checklist.get("seismic_check")),
                ("Flood history", checklist.get("flood_history")),
                ("Environmental clearance", checklist.get("environment_clearance")),
                ("Model verified", checklist.get("model_verified")),
                ("Data freshness", checklist.get("data_freshness")),
                ("Field tests", checklist.get("field_tests")),
                ("Maps verified", checklist.get("maps_verified")),
                ("License verified", checklist.get("license_verified")),
            ]
            for label, ok in items:
                status = "Yes" if ok else "No"
                c.drawString(50, y, f"{label}: {status}")
                y -= 12
            y -= 10

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Engineer Notes")
    y -= 16
    c.setFont("Helvetica", 9)
    c.drawString(50, y, "This report is decision-support only and must be validated with field surveys.")

    c.showPage()
    c.save()
    buf.seek(0)
    return buf

def _land_status(lat, lon):
    """Return 'land', 'water', or 'unknown' based on OSM signals."""
    water_types = {
        "water", "bay", "river", "sea", "ocean", "lake", "reservoir",
        "wetland", "harbour", "lagoon", "stream", "canal", "dam"
    }
    water_keywords = [
        "water", "river", "sea", "ocean", "lake", "reservoir", "wetland",
        "lagoon", "harbour", "stream", "canal", "dam"
    ]
    try:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {
            "format": "jsonv2",
            "lat": lat,
            "lon": lon,
            "zoom": 14,
            "addressdetails": 1
        }
        headers = {"User-Agent": "construction-site-selector/1.0"}
        r = requests.get(url, params=params, headers=headers, timeout=10)
        if r.status_code != 200:
            return "unknown"
        data = r.json()
        category = str(data.get("category", "")).lower()
        place_type = str(data.get("type", "")).lower()
        display = str(data.get("display_name", "")).lower()
        address = data.get("address", {}) or {}
        address_str = " ".join([str(v).lower() for v in address.values()])
        country_code = str(address.get("country_code", "")).lower()

        if country_code and country_code != "in":
            return "water"
        if category in {"water", "natural", "waterway", "place"} and place_type in water_types:
            return "water"
        if place_type in water_types:
            return "water"
        if any(k in display for k in water_keywords):
            return "water"
        if any(k in address_str for k in water_keywords):
            return "water"

        overpass_water = _overpass_is_water(lat, lon)
        if overpass_water:
            return "water"

        if country_code == "in":
            return "land"
        return "unknown"
    except Exception:
        return "unknown"

def _overpass_is_water(lat, lon):
    try:
        url = "https://overpass-api.de/api/interpreter"
        query = f"""
[out:json][timeout:15];
(
  way(around:250,{lat},{lon})["natural"="water"];
  relation(around:250,{lat},{lon})["natural"="water"];
  way(around:250,{lat},{lon})["waterway"];
  relation(around:250,{lat},{lon})["waterway"];
  way(around:250,{lat},{lon})["natural"="bay"];
  relation(around:250,{lat},{lon})["natural"="bay"];
);
out center 1;
"""
        r = requests.post(url, data=query, timeout=10)
        if r.status_code != 200:
            return False
        data = r.json()
        return bool(data.get("elements"))
    except Exception:
        return False

if __name__ == "__main__":
    app.run(debug=True, port=5000)