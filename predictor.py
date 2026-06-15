# predictor.py — ML Model Loading + Prediction

import pickle
import requests # pyre-ignore
import numpy as np # pyre-ignore
import pandas as pd # pyre-ignore
import time
import os
import math
from datetime import datetime, timezone

# ── Patch numpy random-state unpickling for cross-version compatibility ──
# Models were pickled with a different numpy version; the random state stored
# inside sklearn estimators is not essential for inference, so we safely
# replace broken state with a fresh default.
import numpy.random._pickle as _nrp
import numpy.random as _nr
import io

_orig_bg_ctor = _nrp.__bit_generator_ctor
def _safe_bg_ctor(bit_gen_name):
    if isinstance(bit_gen_name, type):
        bit_gen_name = bit_gen_name.__name__
    return _orig_bg_ctor(bit_gen_name)
_nrp.__bit_generator_ctor = _safe_bg_ctor

# Patch MT19937 to ignore incompatible state dicts
_MT = _nr.MT19937
_orig_mt_setstate = getattr(_MT, '__setstate__', None)
def _safe_mt_setstate(self, state):
    try:
        if _orig_mt_setstate:
            _orig_mt_setstate(self, state)
        else:
            super(_MT, self).__setstate__(state)
    except (ValueError, TypeError, KeyError):
        # State format mismatch — reinitialise with a fixed seed
        self.__init__(seed=42)
try:
    _MT.__setstate__ = _safe_mt_setstate
except TypeError:
    pass

# Patch numpy.random.RandomState to handle old state tuples
_RS = np.random.RandomState
_orig_rs_setstate = getattr(_RS, '__setstate__', None)
def _safe_rs_setstate(self, state):
    try:
        if _orig_rs_setstate:
            _orig_rs_setstate(self, state)
    except (ValueError, TypeError, KeyError):
        self.__init__()          # reset to default state
try:
    _RS.__setstate__ = _safe_rs_setstate
except TypeError:
    pass

# ── Load All Models ──
BASE = os.path.join(os.path.dirname(__file__), "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BMTPC_LABELS_PATH = os.path.join(DATA_DIR, "bmtpc_failure_labels.csv")

BHUVAN_API_URL = os.getenv("BHUVAN_API_URL", "")
BHUVAN_API_KEY = os.getenv("BHUVAN_API_KEY", "")
CGWB_API_URL = os.getenv("CGWB_API_URL", "")
CGWB_API_KEY = os.getenv("CGWB_API_KEY", "")
GRID_AVG_ENABLED = os.getenv("GRID_AVG_ENABLED", "1").lower() in {"1", "true", "yes"}

def load(fname):
    with open(os.path.join(BASE, fname), "rb") as f:
        return pickle.load(f)

print("🔄 Loading ML models...")
rf_model       = load("model_feasibility_rf.pkl")
xgb_model      = load("model_feasibility_xgb.pkl")
et_model       = load("model_feasibility_et.pkl")
gb_model       = load("model_lifespan_gb.pkl")
success_model  = load("model_success_rf.pkl")
scaler         = load("scaler.pkl")
label_encoders = load("label_encoders.pkl")
feature_list   = load("feature_list.pkl")
ens_weights    = load("ensemble_weights.pkl")
MODEL_VERSION  = "1.0.0"
MODEL_TRAINED_ON = "india_master_dataset.csv"
DEFAULT_HIST_PATH = os.path.join(os.path.dirname(__file__), "data", "historical_data.csv")
FALLBACK_HIST_PATH = os.path.join(os.path.dirname(__file__), "india_master_dataset (1).csv")
HIST_DATA_PATH = DEFAULT_HIST_PATH
_HIST_DF = None
_BMTPC_DF = None
print("✅ All models loaded!")

# ══════════════════════════════════
# API DATA COLLECTION
# ══════════════════════════════════

# ── Soil derivation helpers (match india_soil.py) ──
def _calculate_bearing_capacity(bdod, clay, sand):
    if bdod is None or clay is None:
        return None
    bd = bdod / 100
    if sand and sand > 60:
        return round(bd * 150, 2)
    if clay and clay > 40:
        return round(bd * 60, 2)
    return round(bd * 100, 2)

def _calculate_shrink_swell(clay):
    if clay is None:
        return "Unknown"
    c = clay / 10
    if c > 40:
        return "High"
    if c > 25:
        return "Medium"
    return "Low"

def _calculate_liquefaction(sand, bdod):
    if sand is None or bdod is None:
        return "Unknown"
    s = sand / 10
    bd = bdod / 100
    if s > 60 and bd < 1.4:
        return "High"
    if s > 40 and bd < 1.6:
        return "Medium"
    return "Low"

def _calculate_permeability(sand, clay):
    if sand is None or clay is None:
        return None
    return round(max((sand / 10 * 2.5) - (clay / 10 * 1.2), 0.1), 2)

def _calculate_corrosion(ph):
    if ph is None:
        return "Unknown"
    p = ph / 10
    if p < 5.5:
        return "High"
    if p < 6.5:
        return "Medium"
    return "Low"

def _estimate_water_table(lat, lon):
    if lon > 79.5 or lon < 72.5:
        return "Shallow (1-3m) — High Risk"
    if lat > 28.0:
        return "Medium (3-8m) — Medium Risk"
    if 15.0 < lat < 25.0 and 74.0 < lon < 82.0:
        return "Deep (10-20m) — Low Risk"
    return "Medium-Deep (5-12m) — Low-Medium Risk"

def _grid_offsets_deg():
    # ~11 km grid (0.05 deg approx) around center
    return [-0.05, 0.0, 0.05]

def _average_numeric_dicts(rows):
    if not rows:
        return {}
    sums = {}
    counts = {}
    for row in rows:
        for k, v in row.items(): # pyre-ignore
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                sums[k] = sums.get(k, 0.0) + float(v)
                counts[k] = counts.get(k, 0) + 1
    out = {}
    for k, total in sums.items():
        out[k] = round(total / counts[k], 4) if counts[k] else None # pyre-ignore
    return out

def _grid_sample_9(fetch_fn, lat, lon):
    rows = []
    for dlat in _grid_offsets_deg():
        for dlon in _grid_offsets_deg():
            try:
                row = fetch_fn(lat + dlat, lon + dlon)
                if row:
                    rows.append(row)
            except Exception:
                continue
    return rows

def _load_bmtpc_labels():
    global _BMTPC_DF
    if _BMTPC_DF is not None:
        return _BMTPC_DF
    if not os.path.exists(BMTPC_LABELS_PATH):
        _BMTPC_DF = pd.DataFrame()
        return _BMTPC_DF
    try:
        df = pd.read_csv(BMTPC_LABELS_PATH)
        _BMTPC_DF = df
        return df
    except Exception:
        _BMTPC_DF = pd.DataFrame()
        return _BMTPC_DF

def _get_bmtpc_risk(lat, lon):
    df = _load_bmtpc_labels()
    if df is None or df.empty:
        return {
            "bmtpc_failure_nearest_km": None,
            "bmtpc_failure_count_25km": 0,
            "bmtpc_failure_severity_index": 0,
        }
    lat_col = "lat" if "lat" in df.columns else "latitude" if "latitude" in df.columns else None
    lon_col = "lon" if "lon" in df.columns else "longitude" if "longitude" in df.columns else None
    if not lat_col or not lon_col:
        return {
            "bmtpc_failure_nearest_km": None,
            "bmtpc_failure_count_25km": 0,
            "bmtpc_failure_severity_index": 0,
        }
    sever_col = "severity" if "severity" in df.columns else None
    dists = df.apply(lambda r: _haversine(lat, lon, r[lat_col], r[lon_col]), axis=1)
    nearest_km = float(dists.min()) if len(dists) else None
    count_25km = int((dists <= 25).sum()) if len(dists) else 0
    if sever_col and len(dists):
        near_idx = dists.idxmin()
        severity = df.loc[near_idx, sever_col]
        severity_index = float(severity) if pd.notna(severity) else 0.0
    else:
        severity_index = 0.0
    return {
        "bmtpc_failure_nearest_km": round(nearest_km, 2) if nearest_km is not None else None, # pyre-ignore
        "bmtpc_failure_count_25km": count_25km,
        "bmtpc_failure_severity_index": round(severity_index, 2), # pyre-ignore
    }

def _get_cgwb_water_table(lat, lon):
    if not CGWB_API_URL:
        return None
    try:
        params = {"lat": lat, "lon": lon}
        if CGWB_API_KEY:
            params["api_key"] = CGWB_API_KEY
        r = requests.get(CGWB_API_URL, params=params, timeout=10)
        data = r.json()
        depth = data.get("water_table_depth_m") or data.get("depth_m") or data.get("depth")
        if depth is None:
            return None
        depth = float(depth)
        if depth <= 3:
            risk = "Shallow (0-3m) — High Risk"
        elif depth <= 8:
            risk = "Medium (3-8m) — Medium Risk"
        elif depth <= 15:
            risk = "Deep (8-15m) — Low Risk"
        else:
            risk = "Very Deep (>15m) — Very Low Risk"
        return {"water_table_depth_m": round(depth, 2), "water_table_risk": risk} # pyre-ignore
    except Exception:
        return None

def _calculate_soil_score(row):
    score = 60 # Start higher
    bc = row.get("bearing_capacity_kNm2")
    if bc is not None:
        if bc > 150:
            score += 20
        elif bc > 100:
            score += 10
        elif bc < 60:
            score -= 30 # Stricter penalty
        if bc < 20:
            score = min(score, 25) # Critical failure for near-zero BC
    
    ss = row.get("shrink_swell_risk")
    if ss == "Low":
        score += 15
    elif ss == "High":
        score -= 25
        
    lq = row.get("liquefaction_risk")
    if lq == "Low":
        score += 10
    elif lq == "High":
        score -= 30
        
    return max(0, min(100, round(score, 1)))

def _recommend_foundation(row):
    bc = row.get("bearing_capacity_kNm2") or 0
    ss = row.get("shrink_swell_risk", "")
    lq = row.get("liquefaction_risk", "")
    if lq == "High":
        return "Pile Foundation (Deep)"
    if bc < 60 or ss == "High":
        return "Raft Foundation"
    if bc < 100:
        return "Isolated Footing with RCC"
    if bc >= 150:
        return "Simple Strip Footing"
    return "Isolated Footing"

# ── Climate derivation helpers (match india_climate.py) ──
CYCLONE_PRONE = [
    {"lat": 13.08, "lon": 80.27, "name": "Chennai Coast",      "risk": "High"},
    {"lat": 16.50, "lon": 81.50, "name": "Andhra Coast",       "risk": "High"},
    {"lat": 20.27, "lon": 85.84, "name": "Odisha Coast",       "risk": "Very High"},
    {"lat": 22.57, "lon": 88.36, "name": "West Bengal Coast",  "risk": "High"},
    {"lat": 10.77, "lon": 79.84, "name": "Nagapattinam",       "risk": "High"},
    {"lat": 15.34, "lon": 73.83, "name": "Goa Coast",          "risk": "Medium"},
    {"lat": 19.07, "lon": 72.87, "name": "Mumbai Coast",       "risk": "Medium"},
    {"lat": 23.02, "lon": 72.57, "name": "Gujarat Coast",      "risk": "High"},
    {"lat": 22.30, "lon": 69.66, "name": "Kutch Coast",        "risk": "High"},
]

def _haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def _get_cyclone_risk(lat, lon):
    dists = [_haversine(lat, lon, c["lat"], c["lon"]) for c in CYCLONE_PRONE]
    min_d = round(min(dists), 2) # pyre-ignore
    if min_d < 50:
        return "Very High", min_d
    if min_d < 150:
        return "High", min_d
    if min_d < 300:
        return "Medium", min_d
    return "Low", min_d

def _get_climate_zone(lat, lon):
    if lat > 32:
        return "Alpine/Sub-Alpine"
    if lat > 28:
        return "Semi-Arid" if lon < 76 else "Humid Subtropical"
    if lat > 23:
        if lon < 70:
            return "Arid"
        if lon < 76:
            return "Semi-Arid"
        return "Humid Subtropical"
    if lat > 18:
        return "Semi-Arid" if lon < 74 else "Tropical Wet & Dry"
    if lat > 12:
        return "Tropical Wet & Dry" if lon < 76 else "Tropical Coastal"
    return "Tropical Humid"

def _get_monsoon_intensity(lat, lon, annual_rain):
    if lon < 77 and lat < 15:
        zone = "SW Monsoon Heavy"
        intensity = "Very High" if annual_rain > 2500 else "High"
    elif 8 < lat < 22 and 80 < lon < 88:
        zone = "NE Monsoon"
        intensity = "High" if annual_rain > 1200 else "Medium"
    elif lat > 20 and 86 < lon < 92:
        zone = "Bay of Bengal Monsoon"
        intensity = "Very High" if annual_rain > 2000 else "High"
    elif lat > 25 and lon < 72:
        zone = "Low Monsoon"
        intensity = "Low"
    else:
        zone = "Normal Monsoon"
        intensity = "Medium" if annual_rain > 800 else "Low"
    return zone, intensity

def _estimate_extreme_heat_days(max_temp):
    if max_temp > 48:
        return 60, "Critical"
    if max_temp > 45:
        return 40, "Very High"
    if max_temp > 42:
        return 20, "High"
    if max_temp > 40:
        return 10, "Medium"
    return 0, "Low"

def _get_drought_risk(annual_rain):
    if annual_rain < 400:
        return "Very High"
    if annual_rain < 700:
        return "High"
    if annual_rain < 1000:
        return "Medium"
    return "Low"

def _get_lightning_risk(lat, lon):
    if 20 < lat < 27 and 80 < lon < 90:
        return "Very High"
    if 22 < lat < 26 and 85 < lon < 92:
        return "High"
    if lat > 25 and lon > 88:
        return "High"
    if 15 < lat < 20 and 73 < lon < 80:
        return "Medium"
    return "Low"

def _estimate_fog_days(lat, lon, min_temp):
    if 25 < lat < 32 and 75 < lon < 88:
        return 40, "High"
    if lat > 28 and min_temp < 10:
        return 20, "Medium"
    return 2, "Low"

def _calculate_heat_index(temp, humidity):
    try:
        hi = (-42.379 + 2.04901523 * temp + 10.14333127 * humidity
              - 0.22475541 * temp * humidity - 0.00683783 * temp ** 2
              - 0.05481717 * humidity ** 2 + 0.00122874 * temp ** 2 * humidity
              + 0.00085282 * temp * humidity ** 2 - 0.00000199 * temp ** 2 * humidity ** 2)
        if hi > 54:
            return round(hi, 1), "Extreme Danger" # pyre-ignore
        if hi > 41:
            return round(hi, 1), "Danger" # pyre-ignore
        if hi > 32:
            return round(hi, 1), "Extreme Caution" # pyre-ignore
        return round(hi, 1), "Caution" # pyre-ignore
    except Exception:
        return None, "Unknown"

def _calculate_climate_score(annual_rain, max_wind, humidity, frost_days,
                             max_temp, min_temp, cyclone_risk,
                             monsoon_intensity, drought_risk,
                             extreme_heat_cat, lightning_risk, fog_risk):
    score = 100
    if annual_rain > 3000:
        score -= 20
    elif annual_rain > 2000:
        score -= 10
    elif annual_rain < 300:
        score -= 15
    elif annual_rain < 500:
        score -= 8
    if max_wind > 20:
        score -= 15
    elif max_wind > 15:
        score -= 8
    if humidity > 90:
        score -= 15
    elif humidity > 85:
        score -= 8
    if frost_days > 30:
        score -= 15
    elif frost_days > 10:
        score -= 8
    if max_temp > 48:
        score -= 15
    elif max_temp > 45:
        score -= 8
    if min_temp < 0:
        score -= 10
    elif min_temp < 5:
        score -= 5
    score += {"Low": 0, "Medium": -10, "High": -20, "Very High": -30}.get(cyclone_risk, 0)
    score += {"Low": 0, "Medium": -5, "High": -10, "Very High": -15}.get(monsoon_intensity, 0)
    score += {"Low": 0, "Medium": -5, "High": -10, "Very High": -15}.get(drought_risk, 0)
    score += {"Low": 0, "Medium": -5, "High": -10, "Very High": -15, "Critical": -20}.get(extreme_heat_cat, 0)
    score += {"Low": 0, "Medium": -5, "High": -10, "Very High": -15}.get(lightning_risk, 0)
    score += {"Low": 0, "Medium": -3, "High": -8}.get(fog_risk, 0)
    return max(0, min(100, round(score, 1)))

def _fetch_soil_point_soilgrids(lat, lon):
    try:
        url    = "https://rest.isric.org/soilgrids/v2.0/properties/query"
        params = {
            "lon"     : lon, "lat": lat,
            "property": ["clay","sand","silt","phh2o",
                         "bdod","cec","soc","nitrogen"],
            "depth"   : "0-5cm", "value": "mean"
        }
        r    = requests.get(url, params=params, timeout=10)
        data = r.json()
        row  = {}
        for layer in data["properties"]["layers"]:
            try:
                row[layer["name"]] = layer["depths"][0]["values"]["mean"]
            except:
                row[layer["name"]] = None
        return row
    except Exception as e:
        print(f"Soil API error: {e}")
        return {}

def _fetch_soil_point_bhuvan(lat, lon):
    if not BHUVAN_API_URL:
        return {}
    try:
        params = {"lat": lat, "lon": lon}
        if BHUVAN_API_KEY:
            params["api_key"] = BHUVAN_API_KEY
        r = requests.get(BHUVAN_API_URL, params=params, timeout=10)
        data = r.json()
        if "properties" in data:
            data = data["properties"]
        if "data" in data:
            data = data["data"]
        row = {
            "clay": data.get("clay"),
            "sand": data.get("sand"),
            "silt": data.get("silt"),
            "phh2o": data.get("ph"),
            "bdod": data.get("bdod") or data.get("bulk_density"),
            "cec": data.get("cec"),
            "soc": data.get("soc"),
            "nitrogen": data.get("nitrogen"),
        }
        return row
    except Exception:
        return {}

def get_soil_data(lat, lon):
    try:
        fetcher = _fetch_soil_point_bhuvan if BHUVAN_API_URL else _fetch_soil_point_soilgrids
        if GRID_AVG_ENABLED:
            rows = _grid_sample_9(fetcher, lat, lon)
            base = _average_numeric_dicts(rows)
        else:
            base = fetcher(lat, lon)

        clay  = base.get("clay")
        sand  = base.get("sand")
        ph    = base.get("phh2o") # pyre-ignore
        bdod  = base.get("bdod") # pyre-ignore
        cec   = base.get("cec") # pyre-ignore
        soc   = base.get("soc") # pyre-ignore
        nit   = base.get("nitrogen") # pyre-ignore
        silt  = base.get("silt") # pyre-ignore

        clay_p = round(clay / 10, 1) if clay else None
        sand_p = round(sand / 10, 1) if sand else None
        silt_p = round(silt / 10, 1) if silt else None
        ph_v = round(ph / 10, 1) if ph else None
        bd = round(bdod / 100, 2) if bdod else None

        bc = _calculate_bearing_capacity(bdod, clay, sand)
        ss = _calculate_shrink_swell(clay)
        lq = _calculate_liquefaction(sand, bdod)
        perm = _calculate_permeability(sand, clay)
        wt = _estimate_water_table(lat, lon)
        cr = _calculate_corrosion(ph)

        cgwb = _get_cgwb_water_table(lat, lon)
        if cgwb:
            wt = cgwb.get("water_table_risk", wt)

        row_out = {
            "clay_percent"          : clay_p,
            "sand_percent"          : sand_p,
            "silt_percent"          : silt_p,
            "ph_value"              : ph_v,
            "bulk_density_gcm3"     : bd,
            "cec_cmolkg"            : round(cec / 10, 1) if cec else None,
            "organic_carbon_percent": round(soc / 10, 2) if soc else None,
            "nitrogen_mgkg"         : nit,
            "bearing_capacity_kNm2" : bc,
            "shrink_swell_risk"     : ss,
            "liquefaction_risk"     : lq,
            "permeability_mmhr"     : perm,
            "estimated_water_table" : wt,
            "corrosion_risk"        : cr,
        }
        if cgwb:
            row_out["water_table_depth_m"] = cgwb.get("water_table_depth_m")
            row_out["water_table_risk"] = cgwb.get("water_table_risk")
        row_out["soil_construction_score"] = _calculate_soil_score(row_out)
        row_out["recommended_foundation"] = _recommend_foundation(row_out)
        return row_out
    except Exception as e:
        print(f"Soil API error: {e}")
        return {}

def _fetch_climate_point(lat, lon):
    try:
        url    = "https://power.larc.nasa.gov/api/temporal/climatology/point"
        params = {
            "parameters": "T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,WS10M_MAX,WS10M,RH2M,FROST_DAYS,ALLSKY_SFC_UV_INDEX",
            "community" : "RE",
            "longitude" : lon, "latitude": lat,
            "format"    : "JSON"
        }
        r     = requests.get(url, params=params, timeout=10)
        props = r.json()["properties"]["parameter"]
        annual_rain = round(sum(props["PRECTOTCORR"].values()), 2)
        max_wind = round(max(props["WS10M_MAX"].values()), 2)
        avg_wind = round(sum(props["WS10M"].values()) / 12, 2) # pyre-ignore
        humidity = round(sum(props["RH2M"].values()) / 12, 2) # pyre-ignore
        frost_days = round(sum(props["FROST_DAYS"].values()), 2)
        avg_temp = round(sum(props["T2M"].values()) / 12, 2) # pyre-ignore
        max_temp = round(max(props["T2M_MAX"].values()), 2)
        min_temp = round(min(props["T2M_MIN"].values()), 2)
        max_rain = round(max(props["PRECTOTCORR"].values()), 2)

        uv_vals = props.get("ALLSKY_SFC_UV_INDEX", {})
        avg_uv = round(sum(uv_vals.values()) / len(uv_vals), 2) if uv_vals else None # pyre-ignore

        cyclone_risk, cyclone_dist = _get_cyclone_risk(lat, lon)
        climate_zone = _get_climate_zone(lat, lon)
        monsoon_zone, monsoon_intensity = _get_monsoon_intensity(lat, lon, annual_rain)
        extreme_heat_days, extreme_heat_cat = _estimate_extreme_heat_days(max_temp)
        drought_risk = _get_drought_risk(annual_rain)
        lightning_risk = _get_lightning_risk(lat, lon)
        fog_days, fog_risk = _estimate_fog_days(lat, lon, min_temp)
        heat_index, heat_index_cat = _calculate_heat_index(avg_temp, humidity)

        climate_score = _calculate_climate_score(
            annual_rain, max_wind, humidity, frost_days,
            max_temp, min_temp, cyclone_risk,
            monsoon_intensity, drought_risk,
            extreme_heat_cat, lightning_risk, fog_risk
        )

        return {
            "avg_temp_C": avg_temp,
            "max_temp_C": max_temp,
            "min_temp_C": min_temp,
            "annual_rainfall_mm": annual_rain,
            "max_monthly_rain_mm": max_rain,
            "avg_wind_speed_ms": avg_wind,
            "max_wind_speed_ms": max_wind,
            "avg_humidity_percent": humidity,
            "frost_days_per_year": frost_days,
            "avg_uv_index": avg_uv,
        }
    except Exception as e:
        print(f"Climate API error: {e}")
        return {}

def get_climate_data(lat, lon):
    try:
        if GRID_AVG_ENABLED:
            rows = _grid_sample_9(_fetch_climate_point, lat, lon)
            base = _average_numeric_dicts(rows)
        else:
            base = _fetch_climate_point(lat, lon)

        annual_rain = base.get("annual_rainfall_mm", 1000) # pyre-ignore
        max_wind = base.get("max_wind_speed_ms", 10) # pyre-ignore
        avg_wind = base.get("avg_wind_speed_ms", 8) # pyre-ignore
        humidity = base.get("avg_humidity_percent", 70) # pyre-ignore
        frost_days = base.get("frost_days_per_year", 0) # pyre-ignore
        avg_temp = base.get("avg_temp_C", 26) # pyre-ignore
        max_temp = base.get("max_temp_C", 35) # pyre-ignore
        min_temp = base.get("min_temp_C", 18) # pyre-ignore
        max_rain = base.get("max_monthly_rain_mm", 150) # pyre-ignore
        avg_uv = base.get("avg_uv_index") # pyre-ignore

        cyclone_risk, cyclone_dist = _get_cyclone_risk(lat, lon)
        climate_zone = _get_climate_zone(lat, lon)
        monsoon_zone, monsoon_intensity = _get_monsoon_intensity(lat, lon, annual_rain)
        extreme_heat_days, extreme_heat_cat = _estimate_extreme_heat_days(max_temp)
        drought_risk = _get_drought_risk(annual_rain)
        lightning_risk = _get_lightning_risk(lat, lon)
        fog_days, fog_risk = _estimate_fog_days(lat, lon, min_temp)
        heat_index, heat_index_cat = _calculate_heat_index(avg_temp, humidity)

        climate_score = _calculate_climate_score(
            annual_rain, max_wind, humidity, frost_days,
            max_temp, min_temp, cyclone_risk,
            monsoon_intensity, drought_risk,
            extreme_heat_cat, lightning_risk, fog_risk
        )

        return {
            "avg_temp_C"                : round(float(avg_temp), 2), # pyre-ignore
            "max_temp_C"                : round(float(max_temp), 2), # pyre-ignore
            "min_temp_C"                : round(float(min_temp), 2), # pyre-ignore
            "temp_range_C"              : round(float(max_temp) - float(min_temp), 2), # pyre-ignore
            "annual_rainfall_mm"        : round(float(annual_rain), 2), # pyre-ignore
            "max_monthly_rain_mm"       : round(float(max_rain), 2), # pyre-ignore
            "avg_wind_speed_ms"         : round(float(avg_wind), 2), # pyre-ignore
            "max_wind_speed_ms"         : round(float(max_wind), 2), # pyre-ignore
            "avg_humidity_percent"      : round(float(humidity), 2), # pyre-ignore
            "frost_days_per_year"       : round(float(frost_days), 2), # pyre-ignore
            "avg_uv_index"              : round(float(avg_uv), 2) if avg_uv is not None else None, # pyre-ignore
            "climate_zone"              : climate_zone,
            "cyclone_risk"              : cyclone_risk,
            "nearest_cyclone_zone_km"   : cyclone_dist,
            "monsoon_zone"              : monsoon_zone,
            "monsoon_intensity"         : monsoon_intensity,
            "extreme_heat_days_per_year": extreme_heat_days,
            "extreme_heat_category"     : extreme_heat_cat,
            "drought_risk"              : drought_risk,
            "lightning_risk"            : lightning_risk,
            "estimated_fog_days"        : fog_days,
            "fog_risk"                  : fog_risk,
            "heat_index_C"              : heat_index,
            "heat_index_category"       : heat_index_cat,
            "climate_construction_score": climate_score,
        }
    except Exception as e:
        print(f"Climate API error: {e}")
        return {}

def get_env_data(lat, lon):
    try:
        url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
        params = {
            "format": "geojson",
            "latitude": lat,
            "longitude": lon,
            "maxradiuskm": 200,
            "minmagnitude": 3.0,
            "starttime": "2000-01-01",
            "endtime": "2024-01-01",
            "limit": 100,
        }
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        count = data["metadata"]["count"]
        mags = [f["properties"]["mag"] for f in data["features"] if f["properties"]["mag"]]
        max_mag = round(max(mags), 2) if mags else 0.0
        if max_mag >= 6.0 or count > 50:
            eq_risk = "High"
        elif max_mag >= 4.5 or count > 20:
            eq_risk = "Medium"
        else:
            eq_risk = "Low"
    except Exception:
        count, max_mag, eq_risk = 0, 0.0, "Low"

    def _get_seismic_zone(lat, lon):
        if lat > 34 or (lat > 24 and lon > 91) or (lat < 14 and lon > 92):
            return "Zone V", 5
        if lat > 30 or (lat > 22 and lon > 88) or (20 < lat < 24 and 72 < lon < 75):
            return "Zone IV", 4
        if lat > 22 or (15 < lat < 22 and 73 < lon < 80):
            return "Zone III", 3
        return "Zone II", 2

    INDIA_RIVERS = [
        {"name": "Cauvery", "lat": 11.1, "lon": 78.8},
        {"name": "Krishna", "lat": 16.5, "lon": 80.6},
        {"name": "Godavari", "lat": 17.0, "lon": 81.8},
        {"name": "Tungabhadra", "lat": 15.9, "lon": 76.5},
        {"name": "Periyar", "lat": 10.2, "lon": 76.3},
        {"name": "Vaigai", "lat": 9.9, "lon": 78.1},
        {"name": "Ganga", "lat": 25.4, "lon": 83.0},
        {"name": "Yamuna", "lat": 27.0, "lon": 78.5},
        {"name": "Brahmaputra", "lat": 26.5, "lon": 92.5},
        {"name": "Indus", "lat": 31.5, "lon": 74.0},
        {"name": "Narmada", "lat": 22.5, "lon": 76.0},
        {"name": "Tapti", "lat": 21.2, "lon": 74.5},
        {"name": "Mahanadi", "lat": 20.5, "lon": 83.5},
        {"name": "Damodar", "lat": 23.5, "lon": 87.0},
        {"name": "Sabarmati", "lat": 23.0, "lon": 72.5},
    ]

    dists = [_haversine(lat, lon, r["lat"], r["lon"]) for r in INDIA_RIVERS]
    river_dist = round(min(dists), 2) # pyre-ignore
    coastal = lon > 79.5 or lon < 72.5 or lat < 9.0
    igp_plain = 24 < lat < 28 and 75 < lon < 88
    brahma_plain = lat > 25 and lon > 89
    delta = (16 < lat < 18 and 81 < lon < 83) or (22 < lat < 22.5 and 88 < lon < 89)
    if coastal or igp_plain or brahma_plain or delta or river_dist < 10:
        flood_risk = "High"
    elif river_dist < 30:
        flood_risk = "Medium"
    else:
        flood_risk = "Low"

    def _get_tsunami_risk(lat, lon):
        TSUNAMI_ZONES = [
            {"lat": 13.0, "lon": 80.3, "risk": "High"},
            {"lat": 11.9, "lon": 79.8, "risk": "High"},
            {"lat": 10.8, "lon": 79.8, "risk": "High"},
            {"lat": 8.5, "lon": 77.5, "risk": "Medium"},
            {"lat": 15.0, "lon": 80.0, "risk": "High"},
            {"lat": 20.0, "lon": 86.5, "risk": "High"},
            {"lat": 11.6, "lon": 92.7, "risk": "Very High"},
        ]
        d = [_haversine(lat, lon, t["lat"], t["lon"]) for t in TSUNAMI_ZONES]
        min_d = round(min(d), 2) # pyre-ignore
        if min_d < 20:
            return TSUNAMI_ZONES[d.index(min(d))]["risk"], min_d
        if min_d < 100:
            return "Low", min_d
        return "None", min_d

    def _get_landslide_risk(lat, lon):
        if lat > 30 and 75 < lon < 92:
            return "Very High", 10
        if lat > 22 and lon > 90:
            return "High", 25
        if lon < 77.5 and 8 < lat < 15:
            return "High", 30
        if 15 < lat < 20 and 80 < lon < 83:
            return "Medium", 55
        if 11 < lat < 11.5 and 76.5 < lon < 77:
            return "High", 30
        return "Low", 85

    def _get_coastal_erosion(lat, lon):
        EROSION_HOTSPOTS = [
            {"lat": 13.0, "lon": 80.3, "risk": "High"},
            {"lat": 20.3, "lon": 86.7, "risk": "Very High"},
            {"lat": 22.0, "lon": 88.5, "risk": "Very High"},
            {"lat": 15.5, "lon": 80.0, "risk": "High"},
            {"lat": 10.9, "lon": 79.8, "risk": "High"},
        ]
        is_coastal = lon > 79.0 or lon < 73.0 or lat < 9.0
        if not is_coastal:
            return "None", 999
        d = [_haversine(lat, lon, e["lat"], e["lon"]) for e in EROSION_HOTSPOTS]
        min_d = round(min(d), 2) # pyre-ignore
        if min_d < 30:
            return EROSION_HOTSPOTS[d.index(min(d))]["risk"], min_d
        if min_d < 100:
            return "Low", min_d
        return "None", min_d

    def _get_mining_risk(lat, lon):
        MINING_ZONES = [
            {"lat": 23.8, "lon": 86.4, "name": "Jharia Coalfield", "risk": "Very High"},
            {"lat": 23.5, "lon": 85.3, "name": "Ranchi Mining", "risk": "High"},
            {"lat": 22.0, "lon": 85.8, "name": "Rourkela Steel Zone", "risk": "High"},
            {"lat": 21.2, "lon": 81.6, "name": "Chhattisgarh Coal", "risk": "High"},
            {"lat": 15.3, "lon": 76.9, "name": "Bellary Iron Ore", "risk": "High"},
            {"lat": 22.7, "lon": 86.2, "name": "Singhbhum Copper", "risk": "Medium"},
            {"lat": 14.4, "lon": 78.8, "name": "Kurnool Mines", "risk": "Medium"},
            {"lat": 25.3, "lon": 83.0, "name": "Mirzapur Quarries", "risk": "Medium"},
        ]
        d = [_haversine(lat, lon, m["lat"], m["lon"]) for m in MINING_ZONES]
        min_d = round(min(d), 2) # pyre-ignore
        idx = d.index(min(d))
        if min_d < 10:
            return MINING_ZONES[idx]["risk"], min_d, MINING_ZONES[idx]["name"]
        if min_d < 25:
            return "Medium", min_d, MINING_ZONES[idx]["name"]
        if min_d < 50:
            return "Low", min_d, MINING_ZONES[idx]["name"]
        return "None", min_d, "None"

    def _get_industrial_hazard(lat, lon):
        INDUSTRIAL_ZONES = [
            {"lat": 13.2, "lon": 80.3, "name": "Manali Industrial Chennai", "risk": "High"},
            {"lat": 11.7, "lon": 79.7, "name": "SIPCOT Cuddalore", "risk": "Very High"},
            {"lat": 13.4, "lon": 80.1, "name": "Gummidipoondi SIPCOT", "risk": "High"},
            {"lat": 11.0, "lon": 77.0, "name": "Coimbatore SIDCO", "risk": "Medium"},
            {"lat": 17.4, "lon": 78.5, "name": "Hyderabad Industrial", "risk": "High"},
            {"lat": 12.9, "lon": 77.6, "name": "Bangalore Whitefield", "risk": "Medium"},
            {"lat": 19.1, "lon": 72.9, "name": "Mumbai Thane Industrial", "risk": "Very High"},
            {"lat": 22.3, "lon": 70.8, "name": "Jamnagar Petrochemical", "risk": "Very High"},
            {"lat": 21.2, "lon": 72.8, "name": "Surat Industrial", "risk": "High"},
            {"lat": 28.7, "lon": 77.1, "name": "Delhi NCR Industrial", "risk": "High"},
            {"lat": 27.5, "lon": 77.7, "name": "Mathura Refinery", "risk": "Very High"},
            {"lat": 22.8, "lon": 86.2, "name": "Jamshedpur Steel", "risk": "High"},
            {"lat": 22.6, "lon": 88.4, "name": "Haldia Petrochemical", "risk": "Very High"},
        ]
        d = [_haversine(lat, lon, i["lat"], i["lon"]) for i in INDUSTRIAL_ZONES]
        min_d = round(min(d), 2) # pyre-ignore
        idx = d.index(min(d))
        if min_d < 5:
            return INDUSTRIAL_ZONES[idx]["risk"], min_d, INDUSTRIAL_ZONES[idx]["name"]
        if min_d < 15:
            return "Medium", min_d, INDUSTRIAL_ZONES[idx]["name"]
        if min_d < 30:
            return "Low", min_d, INDUSTRIAL_ZONES[idx]["name"]
        return "None", min_d, "None"

    def _get_wind_zone(lat, lon):
        if lat < 10 or (lat < 15 and lon > 79) or (lat > 20 and lon < 70):
            return "Zone VI", 55, "Very High"
        if (lat < 14 and lon < 77) or (lat > 22 and lon < 72):
            return "Zone V", 50, "High"
        if lat > 28 or (14 < lat < 20 and lon > 80):
            return "Zone IV", 47, "High"
        if 20 < lat < 28:
            return "Zone III", 44, "Medium"
        return "Zone II", 39, "Low"

    def _get_slope_risk(lat, lon):
        if lat > 30 and 75 < lon < 92:
            return "Very Steep (>30°)", "Very High", 5
        if (lon < 77.5 and 8 < lat < 15) or (lat > 25 and lon > 90):
            return "Steep (15-30°)", "High", 20
        if (77.5 < lon < 79 and 10 < lat < 13) or (lat > 22 and lon > 92):
            return "Moderate (5-15°)", "Medium", 55
        if lon > 79.5 or (22 < lat < 28 and 72 < lon < 76):
            return "Flat-Coastal (<2°)", "Low", 75
        return "Gentle (2-5°)", "Low", 85

    def _get_forest_fire_risk(lat, lon):
        if 20 < lat < 25 and 80 < lon < 85:
            return "High"
        if lat > 25 and 72 < lon < 78:
            return "Medium"
        if lon < 77.5 and 8 < lat < 15:
            return "Medium"
        if lat > 26 and lon > 92:
            return "High"
        return "Low"

    def _get_groundwater_depth(lat, lon):
        if lon > 79.5 or lat < 9:
            return "Shallow (0-3m)", "High Risk"
        if lat > 28 and 75 < lon < 85:
            return "Medium (3-8m)", "Medium Risk"
        if 15 < lat < 25 and 74 < lon < 82:
            return "Deep (10-25m)", "Low Risk"
        if lat < 15 and lon < 77:
            return "Deep (15-30m)", "Low Risk"
        return "Medium-Deep (5-15m)", "Low Risk"

    def _get_drainage_quality(lat, lon):
        coastal = lon > 79.5 or lat < 9
        igp = 24 < lat < 28 and 75 < lon < 88
        if coastal or igp:
            return "Poor"
        if lon < 77.5 and 8 < lat < 15:
            return "Good"
        return "Moderate"

    def _get_air_quality_zone(lat, lon):
        if 28 < lat < 29 and 76 < lon < 78:
            return "Critical", "Delhi NCR"
        if 22.4 < lat < 22.7 and 88.2 < lon < 88.5:
            return "Very High", "Kolkata"
        if 19.0 < lat < 19.2 and 72.8 < lon < 73.0:
            return "High", "Mumbai"
        if 23.7 < lat < 23.9 and 86.3 < lon < 86.5:
            return "High", "Jharia Industrial"
        if 13.0 < lat < 13.2 and 80.2 < lon < 80.4:
            return "Medium", "Chennai"
        return "Low", "Clean Zone"

    tsunami_risk, tsunami_dist = _get_tsunami_risk(lat, lon)
    landslide_risk, _ = _get_landslide_risk(lat, lon)
    erosion_risk, erosion_dist = _get_coastal_erosion(lat, lon)
    mining_risk, mine_dist, mine_name = _get_mining_risk(lat, lon)
    ind_risk, ind_dist, ind_name = _get_industrial_hazard(lat, lon)
    wind_zone, wind_speed, wind_sev = _get_wind_zone(lat, lon)
    slope_desc, slope_risk, _ = _get_slope_risk(lat, lon)
    fire_risk = _get_forest_fire_risk(lat, lon)
    gw_depth, gw_risk = _get_groundwater_depth(lat, lon)
    drainage = _get_drainage_quality(lat, lon)
    air_quality, air_zone = _get_air_quality_zone(lat, lon)
    seismic_zone, seismic_num = _get_seismic_zone(lat, lon)

    score = 100
    score += {"Low": 0, "Medium": -10, "High": -20}.get(eq_risk, 0)
    score += {"Low": 0, "Medium": -10, "High": -20}.get(flood_risk, 0)
    score += {"Low": 0, "Medium": -8, "High": -18, "Very High": -25}.get(landslide_risk, 0)
    score += {"None": 0, "Low": -5, "Medium": -12, "High": -20, "Very High": -30}.get(tsunami_risk, 0) # pyre-ignore
    score += {"None": 0, "Low": -3, "Medium": -10, "High": -18, "Very High": -25}.get(erosion_risk, 0) # pyre-ignore
    score += {"None": 0, "Low": -3, "Medium": -8, "High": -15, "Very High": -20}.get(mining_risk, 0) # pyre-ignore
    score += {"None": 0, "Low": -3, "Medium": -8, "High": -15, "Very High": -20}.get(ind_risk, 0) # pyre-ignore
    score += {"Low": 0, "Medium": -5, "High": -10, "Very High": -15}.get(wind_sev, 0)
    score += {"Low": 0, "Medium": -5, "High": -12, "Very High": -20}.get(slope_risk, 0)
    score += {"Low": 0, "Medium": -5, "High": -12}.get(fire_risk, 0)
    score += {"Good": 0, "Moderate": -5, "Poor": -10}.get(drainage, 0)
    score += {"Low": 0, "Medium": -5, "High": -10, "Very High": -15, "Critical": -20}.get(air_quality, 0)
    if seismic_num >= 5:
        score -= 20
    elif seismic_num == 4:
        score -= 10
    elif seismic_num == 3:
        score -= 5

    return {
        "earthquake_count": count,
        "max_earthquake_magnitude": max_mag,
        "earthquake_risk": eq_risk,
        "seismic_zone": seismic_zone,
        "seismic_zone_number": seismic_num,
        "flood_risk": flood_risk,
        "nearest_river_dist_km": river_dist,
        "landslide_risk": landslide_risk,
        "tsunami_risk": tsunami_risk,
        "tsunami_zone_dist_km": tsunami_dist,
        "coastal_erosion_risk": erosion_risk,
        "erosion_zone_dist_km": erosion_dist,
        "mining_subsidence_risk": mining_risk,
        "nearest_mining_km": mine_dist,
        "nearest_mining_zone": mine_name,
        "industrial_hazard_risk": ind_risk,
        "nearest_industrial_km": ind_dist,
        "nearest_industrial_zone": ind_name,
        "wind_zone": wind_zone,
        "basic_wind_speed_ms": wind_speed,
        "wind_severity": wind_sev,
        "terrain_slope": slope_desc,
        "slope_risk": slope_risk,
        "forest_fire_risk": fire_risk,
        "groundwater_depth": gw_depth,
        "groundwater_risk": gw_risk,
        "drainage_quality": drainage,
        "air_quality_zone": air_quality,
        "air_quality_area": air_zone,
        "env_construction_score": max(0, min(100, round(score, 1))),
    }

def get_animal_data(lat, lon):
    def check_protected_area(lat, lon):
        INDIA_PROTECTED = [
            {"name": "Mudumalai NP", "lat": 11.60, "lon": 76.60, "type": "National Park"},
            {"name": "Anamalai Tiger Reserve", "lat": 10.30, "lon": 77.00, "type": "Tiger Reserve"},
            {"name": "Kalakkad Mundanthurai", "lat": 8.70, "lon": 77.30, "type": "Tiger Reserve"},
            {"name": "Sathyamangalam TR", "lat": 11.50, "lon": 77.20, "type": "Tiger Reserve"},
            {"name": "Guindy NP", "lat": 13.00, "lon": 80.20, "type": "National Park"},
            {"name": "Gulf of Mannar Marine", "lat": 9.10, "lon": 79.10, "type": "Marine Park"},
            {"name": "Nagarhole NP", "lat": 12.10, "lon": 76.10, "type": "National Park"},
            {"name": "Bandipur NP", "lat": 11.70, "lon": 76.60, "type": "National Park"},
            {"name": "Bhadra TR", "lat": 13.50, "lon": 75.60, "type": "Tiger Reserve"},
            {"name": "Periyar TR", "lat": 9.50, "lon": 77.20, "type": "Tiger Reserve"},
            {"name": "Silent Valley NP", "lat": 11.10, "lon": 76.50, "type": "National Park"},
            {"name": "Eravikulam NP", "lat": 10.10, "lon": 77.10, "type": "National Park"},
            {"name": "Kanha TR", "lat": 22.30, "lon": 80.60, "type": "Tiger Reserve"},
            {"name": "Pench TR", "lat": 21.70, "lon": 79.30, "type": "Tiger Reserve"},
            {"name": "Satpura NP", "lat": 22.50, "lon": 78.30, "type": "National Park"},
            {"name": "Bandhavgarh TR", "lat": 23.70, "lon": 81.00, "type": "Tiger Reserve"},
            {"name": "Panna TR", "lat": 24.70, "lon": 80.00, "type": "Tiger Reserve"},
            {"name": "Tadoba TR", "lat": 20.20, "lon": 79.30, "type": "Tiger Reserve"},
            {"name": "Jim Corbett NP", "lat": 29.50, "lon": 78.80, "type": "National Park"},
            {"name": "Rajaji NP", "lat": 30.00, "lon": 78.20, "type": "National Park"},
            {"name": "Kaziranga NP", "lat": 26.60, "lon": 93.40, "type": "National Park"},
            {"name": "Manas NP", "lat": 26.70, "lon": 90.70, "type": "National Park"},
            {"name": "Sundarbans TR", "lat": 21.90, "lon": 89.00, "type": "Tiger Reserve"},
            {"name": "Ranthambore TR", "lat": 26.00, "lon": 76.50, "type": "Tiger Reserve"},
            {"name": "Sariska TR", "lat": 27.30, "lon": 76.40, "type": "Tiger Reserve"},
            {"name": "Gir NP", "lat": 21.10, "lon": 70.80, "type": "National Park"},
            {"name": "Blackbuck NP Velavadar", "lat": 22.00, "lon": 72.20, "type": "National Park"},
            {"name": "Marine NP Jamnagar", "lat": 22.50, "lon": 70.50, "type": "Marine Park"},
            {"name": "Namdapha NP", "lat": 27.50, "lon": 96.40, "type": "National Park"},
            {"name": "Dibru Saikhowa NP", "lat": 27.50, "lon": 95.30, "type": "National Park"},
        ]
        dists = [_haversine(lat, lon, p["lat"], p["lon"]) for p in INDIA_PROTECTED]
        min_d = round(min(dists), 2) # pyre-ignore
        idx = dists.index(min(dists))
        ptype = INDIA_PROTECTED[idx]["type"]
        pname = INDIA_PROTECTED[idx]["name"]
        if min_d < 5:
            risk = "Very High"
        elif min_d < 15:
            risk = "High"
        elif min_d < 30:
            risk = "Medium"
        else:
            risk = "Low"
        return min_d, pname, ptype, risk

    TIGER_CORRIDORS = [
        {"name": "Nilgiris-Eastern Ghats", "lat": 11.40, "lon": 76.80},
        {"name": "Anamalai-Parambikulam", "lat": 10.40, "lon": 77.10},
        {"name": "Central Indian Corridor", "lat": 22.00, "lon": 79.50},
        {"name": "Terai Arc Landscape", "lat": 29.00, "lon": 79.50},
        {"name": "Sundarbans Corridor", "lat": 22.00, "lon": 88.80},
        {"name": "NE Hills Corridor", "lat": 26.50, "lon": 93.00},
        {"name": "Western Ghats Corridor", "lat": 11.00, "lon": 76.50},
        {"name": "Satpura-Maikal Corridor", "lat": 22.50, "lon": 80.00},
        {"name": "Panna-Pench Corridor", "lat": 23.00, "lon": 79.80},
        {"name": "Ranthambore Corridor", "lat": 26.20, "lon": 76.80},
    ]

    def _check_corridor(lat, lon, corridors, high=8, mid=20, low=40):
        d = [_haversine(lat, lon, c["lat"], c["lon"]) for c in corridors]
        min_d = round(min(d), 2) # pyre-ignore
        name = corridors[d.index(min(d))]["name"]
        if min_d < high:
            risk = "Very High"
        elif min_d < mid:
            risk = "High"
        elif min_d < low:
            risk = "Medium"
        else:
            risk = "Low"
        return min_d, name, risk

    ELEPHANT_CORRIDORS = [
        {"name": "Nilgiris Corridor", "lat": 11.40, "lon": 76.80},
        {"name": "Anamalai Corridor", "lat": 10.40, "lon": 77.10},
        {"name": "Sathyamangalam Corridor", "lat": 11.60, "lon": 77.30},
        {"name": "Kalakkad Corridor", "lat": 8.80, "lon": 77.40},
        {"name": "Mudumalai-Bandipur", "lat": 11.65, "lon": 76.65},
        {"name": "Assam Elephant Corridor", "lat": 26.80, "lon": 93.50},
        {"name": "Jharkhand Elephant Belt", "lat": 23.50, "lon": 85.50},
        {"name": "Odisha Elephant Corridor", "lat": 21.50, "lon": 84.50},
        {"name": "North Bengal Corridor", "lat": 26.80, "lon": 89.00},
        {"name": "Eastern Ghats Corridor", "lat": 18.00, "lon": 83.00},
    ]

    BIRD_ZONES = [
        {"name": "Bharatpur Keoladeo", "lat": 27.20, "lon": 77.50},
        {"name": "Chilika Lake", "lat": 19.70, "lon": 85.30},
        {"name": "Point Calimere", "lat": 10.30, "lon": 79.80},
        {"name": "Vedanthangal", "lat": 12.50, "lon": 79.90},
        {"name": "Pulicat Lake", "lat": 13.50, "lon": 80.20},
        {"name": "Koonthankulam", "lat": 8.80, "lon": 77.70},
        {"name": "Nal Sarovar Gujarat", "lat": 22.80, "lon": 72.00},
        {"name": "Loktak Lake Manipur", "lat": 24.50, "lon": 93.80},
        {"name": "Sambhar Lake Rajasthan", "lat": 26.90, "lon": 75.10},
        {"name": "Harike Wetland Punjab", "lat": 31.20, "lon": 75.20},
        {"name": "Pichavaram Mangrove", "lat": 11.40, "lon": 79.80},
        {"name": "Rann of Kutch Staging", "lat": 23.80, "lon": 69.80},
        {"name": "Gujarat Coast Flyway", "lat": 22.00, "lon": 72.00},
        {"name": "TN Coast Flyway", "lat": 12.00, "lon": 80.10},
    ]

    ENDANGERED_HABITATS = [
        {"name": "Nilgiri Tahr", "lat": 10.10, "lon": 77.20, "species": "Nilgiri Tahr"},
        {"name": "Lion-tailed Macaque", "lat": 10.30, "lon": 77.00, "species": "Lion-tailed Macaque"},
        {"name": "Asiatic Lion Gir", "lat": 21.10, "lon": 70.80, "species": "Asiatic Lion"},
        {"name": "One-horned Rhino", "lat": 26.60, "lon": 93.40, "species": "One-horned Rhino"},
        {"name": "Bengal Tiger Core", "lat": 22.30, "lon": 80.60, "species": "Bengal Tiger"},
        {"name": "Snow Leopard Habitat", "lat": 33.00, "lon": 77.50, "species": "Snow Leopard"},
        {"name": "Gangetic Dolphin", "lat": 25.40, "lon": 83.00, "species": "Gangetic Dolphin"},
        {"name": "Dugong Gulf of Mannar", "lat": 9.20, "lon": 79.20, "species": "Dugong"},
        {"name": "Sea Turtle Chennai", "lat": 13.20, "lon": 80.30, "species": "Olive Ridley"},
        {"name": "Great Indian Bustard", "lat": 27.00, "lon": 71.00, "species": "Great Indian Bustard"},
        {"name": "Red Panda NE India", "lat": 27.20, "lon": 88.60, "species": "Red Panda"},
        {"name": "Hangul Kashmir", "lat": 34.00, "lon": 75.00, "species": "Kashmir Stag"},
        {"name": "Irrawaddy Dolphin", "lat": 26.50, "lon": 95.00, "species": "Irrawaddy Dolphin"},
        {"name": "Leatherback Turtle", "lat": 12.00, "lon": 93.00, "species": "Leatherback Turtle"},
        {"name": "Grizzled Squirrel", "lat": 9.50, "lon": 77.50, "species": "Grizzled Squirrel"},
    ]

    CONFLICT_ZONES = [
        {"name": "Nilgiris Fringe", "lat": 11.40, "lon": 76.70},
        {"name": "Coimbatore Forest Border", "lat": 11.10, "lon": 76.90},
        {"name": "Assam Tea Garden Conflicts", "lat": 26.50, "lon": 93.80},
        {"name": "Jharkhand Elephant Conflict", "lat": 23.30, "lon": 85.30},
        {"name": "Odisha Elephant Conflict", "lat": 21.50, "lon": 84.80},
        {"name": "Sundarbans Tiger Conflict", "lat": 21.80, "lon": 89.10},
        {"name": "Uttarakhand Leopard Conflict", "lat": 29.50, "lon": 79.00},
        {"name": "Karnataka Forest Fringe", "lat": 12.30, "lon": 76.40},
        {"name": "Gujarat Lion Fringe", "lat": 21.30, "lon": 70.60},
        {"name": "MP Tiger Conflict Zone", "lat": 22.50, "lon": 80.50},
    ]

    MARINE_ZONES = [
        {"name": "Gulf of Mannar", "lat": 9.10, "lon": 79.10},
        {"name": "Lakshadweep Marine", "lat": 10.60, "lon": 72.60},
        {"name": "Marine NP Jamnagar", "lat": 22.50, "lon": 70.50},
        {"name": "Malvan Marine Sanctuary", "lat": 16.10, "lon": 73.50},
        {"name": "Andaman Marine", "lat": 12.00, "lon": 93.00},
        {"name": "Chilika Marine", "lat": 19.70, "lon": 85.30},
        {"name": "Mangrove Bhitarkanika", "lat": 20.70, "lon": 87.00},
        {"name": "Pichavaram Mangrove TN", "lat": 11.40, "lon": 79.80},
    ]

    def check_bird_zone(lat, lon):
        d = [_haversine(lat, lon, b["lat"], b["lon"]) for b in BIRD_ZONES]
        min_d = round(min(d), 2) # pyre-ignore
        name = BIRD_ZONES[d.index(min(d))]["name"]
        if min_d < 5:
            risk = "High"
        elif min_d < 15:
            risk = "Medium"
        else:
            risk = "Low"
        return min_d, name, risk

    def check_endangered_habitat(lat, lon):
        d = [_haversine(lat, lon, e["lat"], e["lon"]) for e in ENDANGERED_HABITATS]
        min_d = round(min(d), 2) # pyre-ignore
        idx = d.index(min(d))
        species = ENDANGERED_HABITATS[idx]["species"]
        name = ENDANGERED_HABITATS[idx]["name"]
        if min_d < 10:
            risk = "Very High"
        elif min_d < 25:
            risk = "High"
        elif min_d < 50:
            risk = "Medium"
        else:
            risk = "Low"
        return min_d, species, name, risk

    def check_conflict_zone(lat, lon):
        d = [_haversine(lat, lon, c["lat"], c["lon"]) for c in CONFLICT_ZONES]
        min_d = round(min(d), 2) # pyre-ignore
        name = CONFLICT_ZONES[d.index(min(d))]["name"]
        if min_d < 10:
            risk = "High"
        elif min_d < 25:
            risk = "Medium"
        else:
            risk = "Low"
        return min_d, name, risk

    def check_marine_zone(lat, lon):
        coastal = lon > 79.0 or lon < 73.5 or lat < 9.0
        if not coastal:
            return "None", 999, "Inland"
        d = [_haversine(lat, lon, m["lat"], m["lon"]) for m in MARINE_ZONES]
        min_d = round(min(d), 2) # pyre-ignore
        idx = d.index(min(d))
        name = MARINE_ZONES[idx]["name"]
        if min_d < 5:
            risk = "Very High"
        elif min_d < 20:
            risk = "High"
        elif min_d < 50:
            risk = "Medium"
        else:
            risk = "Low"
        return risk, min_d, name

    def check_biodiversity_hotspot(lat, lon):
        if lon < 78.0 and 8.0 < lat < 21.0:
            return "Yes", "Western Ghats (UNESCO)"
        if lat > 26.0 and lon > 88.0:
            return "Yes", "Eastern Himalayas (UNESCO)"
        if lat > 22.0 and lon > 92.0:
            return "Yes", "Indo-Burma Hotspot"
        if 8.5 < lat < 10.5 and 78.5 < lon < 79.8:
            return "Yes", "Gulf of Mannar"
        if lat < 14.0 and lon > 92.0:
            return "Yes", "Sundaland (Andaman)"
        return "No", "None"

    def get_burrowing_risk(lat, lon, clay_percent=None):
        near_forest = lon < 78.5 and 8 < lat < 25
        high_clay = clay_percent and clay_percent > 30
        arid_zone = lat > 24 and lon < 73
        if arid_zone:
            return "High"
        if near_forest and high_clay:
            return "High"
        if near_forest or high_clay:
            return "Medium"
        return "Low"

    def get_gbif_data(lat, lon):
        try:
            url = "https://api.gbif.org/v1/occurrence/search"
            params = {
                "decimalLatitude": f"{lat-0.3},{lat+0.3}",
                "decimalLongitude": f"{lon-0.3},{lon+0.3}",
                "kingdomKey": 1,
                "hasCoordinate": True,
                "limit": 100,
            }
            r = requests.get(url, params=params, timeout=20)
            data = r.json()
            records = data.get("results", [])
            total = data.get("count", 0)
            threatened = len([x for x in records if x.get("iucnRedListCategory") in ["CR", "EN", "VU"]])
            species = len(set([x.get("speciesKey") for x in records if x.get("speciesKey")]))
            mammals = len([x for x in records if x.get("class") == "Mammalia"])
            birds = len([x for x in records if x.get("class") == "Aves"])
            return total, threatened, species, mammals, birds
        except Exception:
            return 0, 0, 0, 0, 0

    def calculate_animal_score(pa_risk, tiger_risk, elephant_risk,
                               bird_risk, endangered_risk, conflict_risk,
                               marine_risk, is_hotspot, burrowing_risk,
                               threatened_count):
        score = 100
        score += {"Low": 0, "Medium": -15, "High": -25, "Very High": -40}.get(pa_risk, 0)
        score += {"Low": 0, "Medium": -10, "High": -20, "Very High": -30}.get(tiger_risk, 0)
        score += {"Low": 0, "Medium": -10, "High": -20, "Very High": -25}.get(elephant_risk, 0)
        score += {"Low": 0, "Medium": -5, "High": -12}.get(bird_risk, 0)
        score += {"Low": 0, "Medium": -8, "High": -15, "Very High": -20}.get(endangered_risk, 0)
        score += {"Low": 0, "Medium": -5, "High": -15}.get(conflict_risk, 0)
        score += {"None": 0, "Low": -3, "Medium": -10, "High": -18, "Very High": -25}.get(marine_risk, 0)
        if is_hotspot == "Yes":
            score -= 10
        score += {"Low": 0, "Medium": -3, "High": -8}.get(burrowing_risk, 0)
        if threatened_count > 5:
            score -= 10
        elif threatened_count > 2:
            score -= 5
        return max(0, min(100, round(score, 1)))

    def get_building_success_label(lat, lon):
        score = 0
        weight = 0
        try:
            url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
            params = {
                "format": "geojson",
                "latitude": lat,
                "longitude": lon,
                "maxradiuskm": 150,
                "minmagnitude": 5.0,
                "starttime": "1990-01-01",
                "endtime": "2024-01-01",
            }
            r = requests.get(url, params=params, timeout=20)
            data = r.json()
            mags = [f["properties"]["mag"] for f in data["features"] if f["properties"]["mag"]]
            max_mag = max(mags) if mags else 0
            count = len(mags)
            if max_mag >= 7.0 or count > 20:
                usgs_score = 0
            elif max_mag >= 6.0 or count > 10:
                usgs_score = 30
            elif max_mag >= 5.0:
                usgs_score = 60
            else:
                usgs_score = 90
            score += usgs_score * 0.35
            weight += 0.35
        except Exception:
            score += 60 * 0.35
            weight += 0.35

        MAJOR_CITIES = [
            (28.67, 77.21), (19.08, 72.88), (12.97, 77.59),
            (22.57, 88.36), (17.38, 78.48), (13.08, 80.27),
            (23.02, 72.57), (26.91, 75.79), (21.25, 81.63),
            (11.01, 76.96), (9.93, 78.12), (10.79, 78.70),
            (15.34, 75.14), (16.51, 80.64), (20.27, 85.84),
            (25.59, 85.14), (26.85, 80.95), (22.32, 87.32),
        ]
        city_dists = [_haversine(lat, lon, c[0], c[1]) for c in MAJOR_CITIES]
        min_city_d = min(city_dists)
        if min_city_d < 10:
            osm_score = 95
        elif min_city_d < 30:
            osm_score = 80
        elif min_city_d < 60:
            osm_score = 65
        elif min_city_d < 100:
            osm_score = 55
        else:
            osm_score = 45
        score += osm_score * 0.35
        weight += 0.35

        geo_score = 70
        if 15 < lat < 28 and 74 < lon < 85:
            geo_score += 15
        if lon > 79.5 or lon < 72.5:
            geo_score -= 20
        if lat > 32:
            geo_score -= 35
        if 24 < lat < 28 and 80 < lon < 88:
            geo_score -= 10
        if lon < 77.5 and 8 < lat < 15:
            geo_score -= 15
        if lat > 30 or (lat > 24 and lon > 90):
            geo_score -= 20
        geo_score = max(0, min(100, geo_score))
        score += geo_score * 0.30
        weight += 0.30

        final_score = round(score / weight, 1) if weight > 0 else 50 # pyre-ignore
        if final_score >= 70:
            return 1, "Success", round(final_score, 1)
        if final_score >= 45:
            return 0.5, "Moderate", round(final_score, 1)
        return 0, "High Risk", round(final_score, 1)

    pa_dist, pa_name, pa_type, pa_risk = check_protected_area(lat, lon)
    ti_dist, ti_name, ti_risk = _check_corridor(lat, lon, TIGER_CORRIDORS)
    el_dist, el_name, el_risk = _check_corridor(lat, lon, ELEPHANT_CORRIDORS, high=10, mid=25, low=50)
    bi_dist, bi_name, bi_risk = check_bird_zone(lat, lon)
    en_dist, en_species, en_name, en_risk = check_endangered_habitat(lat, lon)
    cf_dist, cf_name, cf_risk = check_conflict_zone(lat, lon)
    ma_risk, ma_dist, ma_name = check_marine_zone(lat, lon)
    is_hotspot, hotspot_name = check_biodiversity_hotspot(lat, lon)
    burrowing_risk = get_burrowing_risk(lat, lon)
    total, threatened, species, mammals, birds = get_gbif_data(lat, lon)

    animal_score = calculate_animal_score(
        pa_risk, ti_risk, el_risk, bi_risk,
        en_risk, cf_risk, ma_risk,
        is_hotspot, burrowing_risk, threatened
    )
    b_label, b_category, b_score = get_building_success_label(lat, lon)

    return {
        "total_animal_records": total,
        "threatened_species_count": threatened,
        "unique_species_count": species,
        "mammal_count": mammals,
        "bird_count": birds,
        "nearest_protected_area_km": pa_dist,
        "nearest_protected_area": pa_name,
        "protected_area_type": pa_type,
        "protected_area_risk": pa_risk,
        "nearest_tiger_corridor_km": ti_dist,
        "nearest_tiger_corridor": ti_name,
        "tiger_corridor_risk": ti_risk,
        "nearest_elephant_corridor_km": el_dist,
        "nearest_elephant_corridor": el_name,
        "elephant_corridor_risk": el_risk,
        "nearest_bird_zone_km": bi_dist,
        "nearest_bird_zone": bi_name,
        "bird_zone_risk": bi_risk,
        "nearest_endangered_habitat_km": en_dist,
        "nearest_endangered_species": en_species,
        "endangered_habitat_name": en_name,
        "endangered_habitat_risk": en_risk,
        "nearest_conflict_zone_km": cf_dist,
        "nearest_conflict_zone": cf_name,
        "human_animal_conflict_risk": cf_risk,
        "marine_protected_area_risk": ma_risk,
        "nearest_marine_area_km": ma_dist,
        "nearest_marine_area": ma_name,
        "biodiversity_hotspot": is_hotspot,
        "hotspot_name": hotspot_name,
        "burrowing_animal_risk": burrowing_risk,
        "animal_construction_score": animal_score,
        "construction_success_label": b_label,
        "construction_success_category": b_category,
        "construction_viability_score": b_score,
    }

# ══════════════════════════════════
# FEATURE ENRICHMENT
# ══════════════════════════════════
def _haversine_km(la1, lo1, la2, lo2):
    import math
    R = 6371
    d1 = math.radians(la2 - la1)
    d2 = math.radians(lo2 - lo1)
    a = math.sin(d1 / 2) ** 2 + math.cos(math.radians(la1)) * math.cos(math.radians(la2)) * math.sin(d2 / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def _load_historical_data():
    global _HIST_DF
    if _HIST_DF is not None:
        return _HIST_DF
    hist_path = get_hist_path()
    if not os.path.exists(hist_path):
        _HIST_DF = None
        return None
    try:
        df = pd.read_csv(hist_path)
        df.columns = [c.strip() for c in df.columns]
        _HIST_DF = df
        return df
    except Exception as e:
        print(f"Historical dataset load error: {e}")
        _HIST_DF = None
        return None

def get_hist_path():
    if os.path.exists(DEFAULT_HIST_PATH):
        return DEFAULT_HIST_PATH
    if os.path.exists(FALLBACK_HIST_PATH):
        return FALLBACK_HIST_PATH
    return DEFAULT_HIST_PATH

def _get_historical_overrides(lat, lon, max_km=10):
    df = _load_historical_data()
    if df is None or df.empty:
        return {}, None

    lat_col = "latitude" if "latitude" in df.columns else "lat" if "lat" in df.columns else None # pyre-ignore
    lon_col = "longitude" if "longitude" in df.columns else "lon" if "lon" in df.columns else None # pyre-ignore
    if not lat_col or not lon_col:
        return {}, None

    try:
        distances = df.apply(lambda r: _haversine_km(lat, lon, float(r[lat_col]), float(r[lon_col])), axis=1) # pyre-ignore
        idx = distances.idxmin()
        dist_km = float(distances.loc[idx])
        if dist_km > max_km:
            return {}, None
        row = df.loc[idx].to_dict() # pyre-ignore
        row.pop(lat_col, None)
        row.pop(lon_col, None)
        return row, dist_km
    except Exception as e:
        print(f"Historical match error: {e}")
        return {}, None

def _nearest_city_km(lat, lon):
    cities = [
        (28.6139, 77.2090),  # Delhi
        (19.0760, 72.8777),  # Mumbai
        (13.0827, 80.2707),  # Chennai
        (22.5726, 88.3639),  # Kolkata
        (12.9716, 77.5946),  # Bengaluru
        (17.3850, 78.4867),  # Hyderabad
    ]
    return min([_haversine_km(lat, lon, c[0], c[1]) for c in cities])

def _is_coastal(lat, lon):
    return (lon < 73.5) or (lon > 83.5) or (lat < 12.5 and 74 <= lon <= 80)

def _enrich_features(combined, lat, lon):
    annual_rain = combined.get("annual_rainfall_mm", 1000) or 1000
    max_wind = combined.get("max_wind_speed_ms", 10) or 10
    max_temp = combined.get("max_temp_C", 35) or 35
    avg_hum = combined.get("avg_humidity_percent", 70) or 70
    perm = combined.get("permeability_mmhr", 5) or 5
    landslide = combined.get("landslide_risk", "Low")
    protected_risk = combined.get("protected_area_risk", "Low")
    biodiv = combined.get("biodiversity_hotspot", "No")
    animal_records = combined.get("total_animal_records", 0) or 0
    threatened = combined.get("threatened_species_count", 0) or 0

    coastal = _is_coastal(lat, lon)
    city_dist = _nearest_city_km(lat, lon)

    if city_dist < 35:
        air_quality = "High"
        industrial = "High"
    elif city_dist < 90:
        air_quality = "Medium"
        industrial = "Medium"
    else:
        air_quality = "Low"
        industrial = "Low"

    if annual_rain < 600:
        drought = "High"
    elif annual_rain < 900:
        drought = "Medium"
    else:
        drought = "Low"

    if (lat > 25 and avg_hum > 80) or (lat > 28 and annual_rain > 900):
        fog = "High"
    elif lat > 24:
        fog = "Medium"
    else:
        fog = "Low"

    if perm < 5 or annual_rain > 2500:
        drainage = "Poor"
    elif perm < 12 or annual_rain > 1800:
        drainage = "Moderate"
    else:
        drainage = "Good"

    if landslide in ["High", "Very High"]:
        slope = "High"
    elif lat > 25 and 74 < lon < 80:
        slope = "Medium"
    else:
        slope = "Low"

    if coastal and lon > 80 and 8 < lat < 22:
        cyclone = "High"
    elif coastal:
        cyclone = "Medium"
    else:
        cyclone = "Low"

    if coastal and lat < 15 and lon > 80:
        tsunami = "High"
    elif coastal:
        tsunami = "Medium"
    else:
        tsunami = "Low"

    if coastal and (max_wind > 18 or annual_rain > 2000):
        coastal_erosion = "High"
    elif coastal:
        coastal_erosion = "Medium"
    else:
        coastal_erosion = "Low"

    if coastal:
        marine_pa = "Medium" if protected_risk in ["High", "Very High"] else "Low"
    else:
        marine_pa = "Low"

    if biodiv == "Yes" or protected_risk in ["High", "Very High"]:
        endangered = "High"
    elif protected_risk == "Medium":
        endangered = "Medium"
    else:
        endangered = "Low"

    if protected_risk in ["High", "Very High"] and 20 < lat < 28 and 75 < lon < 88:
        tiger_corridor = "High"
    elif protected_risk in ["Medium", "High", "Very High"]:
        tiger_corridor = "Medium"
    else:
        tiger_corridor = "Low"

    if protected_risk in ["High", "Very High"] or animal_records > 1000 or threatened > 3:
        human_conflict = "High"
    elif animal_records > 300 or threatened > 1:
        human_conflict = "Medium"
    else:
        human_conflict = "Low"

    if annual_rain < 700 and max_temp > 38:
        forest_fire = "High"
    elif annual_rain < 1100 and max_temp > 35:
        forest_fire = "Medium"
    else:
        forest_fire = "Low"

    if max_temp > 45:
        extreme_heat_days = 30
    elif max_temp > 40:
        extreme_heat_days = 15
    elif max_temp > 35:
        extreme_heat_days = 5
    else:
        extreme_heat_days = 1

    # Climate zones and derived fields
    if lat < 15:
        climate_zone = "Tropical"
    elif lat < 23:
        climate_zone = "Subtropical"
    else:
        climate_zone = "Temperate"

    monsoon_zone = "High" if lon > 76 and lat < 25 else "Medium" if lat < 28 else "Low"
    monsoon_intensity = "High" if annual_rain > 2000 else "Medium" if annual_rain > 1200 else "Low"
    lightning_risk = "High" if annual_rain > 1800 else "Medium" if annual_rain > 1000 else "Low"
    fog_days = 25 if fog == "High" else 10 if fog == "Medium" else 3
    heat_index = round(max_temp + (avg_hum/100)*5, 1) # pyre-ignore
    heat_index_cat = "Extreme" if heat_index > 46 else "High" if heat_index > 40 else "Moderate"
    extreme_heat_cat = "Extreme" if max_temp > 46 else "High" if max_temp > 40 else "Moderate"

    for key, value in {
        "air_quality_zone": air_quality,
        "industrial_hazard_risk": industrial,
        "drought_risk": drought,
        "fog_risk": fog,
        "drainage_quality": drainage,
        "slope_risk": slope,
        "cyclone_risk": cyclone,
        "tsunami_risk": tsunami,
        "coastal_erosion_risk": coastal_erosion,
        "marine_protected_area_risk": marine_pa,
        "endangered_habitat_risk": endangered,
        "tiger_corridor_risk": tiger_corridor,
        "human_animal_conflict_risk": human_conflict,
        "forest_fire_risk": forest_fire,
        "extreme_heat_days_per_year": extreme_heat_days,
        "climate_zone": climate_zone,
        "monsoon_zone": monsoon_zone,
        "monsoon_intensity": monsoon_intensity,
        "lightning_risk": lightning_risk,
        "estimated_fog_days": fog_days,
        "heat_index_C": heat_index,
        "heat_index_category": heat_index_cat,
        "extreme_heat_category": extreme_heat_cat,
    }.items():
        combined.setdefault(key, value)

    # Environment-only fields (best-effort heuristics)
    combined.setdefault("tsunami_zone_dist_km", 20 if coastal else 200)
    combined.setdefault("erosion_zone_dist_km", 15 if coastal else 180)
    combined.setdefault("mining_subsidence_risk", "Medium" if city_dist < 120 else "Low")
    combined.setdefault("nearest_mining_km", round(city_dist + 30, 1)) # pyre-ignore
    combined.setdefault("nearest_mining_zone", "Central Belt" if city_dist < 150 else "None")
    combined.setdefault("nearest_industrial_km", round(city_dist, 1)) # pyre-ignore
    combined.setdefault("nearest_industrial_zone", "Metro Cluster" if city_dist < 50 else "Regional Cluster")
    combined.setdefault("air_quality_area", "Urban" if city_dist < 50 else "Peri-Urban" if city_dist < 120 else "Rural")
    combined.setdefault("terrain_slope", "Steep" if slope == "High" else "Moderate" if slope == "Medium" else "Gentle")
    combined.setdefault("groundwater_depth", 8.0 if lat < 12 else 12.0 if lat < 20 else 16.0)
    combined.setdefault("groundwater_risk", "High" if lat < 12 else "Medium" if lat < 20 else "Low")


def _build_safety_notes(soil, climate, env, animal):
    notes = []
    river = env.get("nearest_river_dist_km")
    if river is not None:
        fr = env.get("flood_risk", "Low")
        notes.append(f"Nearest river: {river}km — {fr} flood risk")
    sz = env.get("seismic_zone", "")
    if sz:
        notes.append(f"Seismic zone: {sz}")
    pa = animal.get("protected_area_risk", "Low")
    if pa in ["Low", "Medium"]:
        notes.append("No protected area restriction flagged")
    else:
        notes.append(f"Protected area risk: {pa}")
    bc = soil.get("bearing_capacity_kNm2")
    if bc:
        notes.append(f"Soil bearing capacity: {round(float(bc),1)} kN/m²") # pyre-ignore
    return [n for i, n in enumerate(notes) if i < 5]

# ══════════════════════════════════
# MAIN PREDICT FUNCTION
# ══════════════════════════════════
def predict_location(lat, lon, building_type="House", floors=2, sensor_data=None):

    print(f"🔍 Analyzing: {lat}, {lon}")

    # 1. Collect all API data
    fetched_at = datetime.utcnow().isoformat() + "Z"
    
    t0 = time.time()
    soil    = get_soil_data(lat, lon);    print(f"⏱️ Soil data: {time.time()-t0:.2f}s")
    
    t1 = time.time()
    climate = get_climate_data(lat, lon); print(f"⏱️ Climate data: {time.time()-t1:.2f}s")
    
    t2 = time.time()
    env     = get_env_data(lat, lon);     print(f"⏱️ Env data: {time.time()-t2:.2f}s")
    
    t3 = time.time()
    animal  = get_animal_data(lat, lon);   print(f"⏱️ Animal data: {time.time()-t3:.2f}s")

    sensor_data = sensor_data or {}
    local_sensors_used = bool(sensor_data)

    hist_row, hist_dist = _get_historical_overrides(lat, lon)

    bmtpc_risk = _get_bmtpc_risk(lat, lon)
    source_status = {
        "bhuvan": "ok" if BHUVAN_API_URL else "none",
        "soilgrids": "ok" if soil else "fallback",
        "cgwb": "ok" if CGWB_API_URL else "none",
        "nasa_power": "ok" if climate else "fallback",
        "usgs": "ok" if env else "fallback",
        "gbif": "ok" if animal else "fallback",
        "bmtpc_labels": "ok" if os.path.exists(BMTPC_LABELS_PATH) else "none",
        "historical_dataset": "ok" if hist_row else "none",
        "local_sensors": "ok" if local_sensors_used else "none",
    }

    # 2. Combine all data
    combined = {}
    combined.update(soil) # pyre-ignore
    combined.update(climate) # pyre-ignore
    combined.update(env) # pyre-ignore
    combined.update(animal) # pyre-ignore
    combined.update(bmtpc_risk) # pyre-ignore

    if hist_row:
        combined.update(hist_row)

    if sensor_data:
        combined.update(sensor_data) # pyre-ignore

    # 3. Domain scores
    soil_score    = soil.get("soil_construction_score", 50)
    climate_score = climate.get("climate_construction_score", 70)
    env_score     = env.get("env_construction_score", 70)
    animal_score  = animal.get("animal_construction_score", 70)

    combined["soil_construction_score"]    = soil_score # pyre-ignore
    combined["climate_construction_score"] = climate_score # pyre-ignore
    combined["env_construction_score"]     = env_score # pyre-ignore
    combined["animal_construction_score"]  = animal_score # pyre-ignore

    # 4. Composite features
    bc = combined.get("bearing_capacity_kNm2", 100) or 100
    combined["soil_stability_index"] = min(100.0, # pyre-ignore
        bc/200*40 +
        (100 - (combined.get("clay_percent",30) or 30))/100*30 +
        (combined.get("sand_percent",40) or 40)/100*30
    )
    combined["climate_stress_index"] = min(100.0, # pyre-ignore
        (combined.get("annual_rainfall_mm",1000) or 1000)/3000*30 +
        (combined.get("max_wind_speed_ms",10) or 10)/50*30 +
        (combined.get("avg_humidity_percent",70) or 70)/100*20 +
        (combined.get("frost_days_per_year",0) or 0)/100*20
    )
    combined["env_danger_index"] = min(100.0, # pyre-ignore
        (combined.get("seismic_zone_number",2) or 2)/5*35 +
        max(0.0, 100.0-(combined.get("nearest_river_dist_km",50) or 50))/100*25 +
        (combined.get("max_earthquake_magnitude",0) or 0)/8*40
    )
    threatened = combined.get("threatened_species_count", 0) or 0
    total_records = combined.get("total_animal_records", 0) or 0
    protected_km = combined.get("nearest_protected_area_km", 100) or 100
    threat_score = min(1.0, threatened / 50.0)
    total_score = min(1.0, math.log1p(total_records) / math.log1p(50000))
    protected_score = min(1.0, max(0.0, (50.0 - protected_km) / 50.0))
    combined["bio_pressure_index"] = round(min(100.0, # pyre-ignore
        threat_score * 40.0 +
        total_score * 30.0 +
        protected_score * 30.0
    ), 1) # pyre-ignore
    combined["soil_climate_interaction"] = combined["soil_stability_index"] * (1 - combined["climate_stress_index"]/100) # pyre-ignore
    combined["env_bio_interaction"]      = combined["env_danger_index"] * (1 + combined["bio_pressure_index"]/100) # pyre-ignore
    combined["overall_risk_composite"]   = ( # pyre-ignore
        combined["climate_stress_index"]*0.25 +
        combined["env_danger_index"]*0.35 +
        combined["bio_pressure_index"]*0.15 +
        (100 - combined["soil_stability_index"])*0.25
    )

    # 4b. Enrich missing model features using geo + climate heuristics
    _enrich_features(combined, lat, lon)

    # 5. Encode categoricals
    for col, le in label_encoders.items():
        if col in combined:
            try:
                combined[col] = le.transform([str(combined[col])])[0] # pyre-ignore
            except:
                combined[col] = 0 # pyre-ignore

    # 6. Build feature vector
    row = {}
    for feat in feature_list:
        val = combined.get(feat, 0)
        row[feat] = float(val) if val is not None else 0.0

    X = pd.DataFrame([row])[feature_list]
    X = X.fillna(0)
    X_sc = scaler.transform(X)

    # 7. Predict
    rf_pred   = float(rf_model.predict(X)[0])
    xgb_pred  = float(xgb_model.predict(X_sc)[0])
    et_pred   = float(et_model.predict(X)[0])
    life_pred = float(gb_model.predict(X)[0])
    succ_pred = float(success_model.predict(X)[0])

    # 8. Weighted ensemble
    w = ens_weights
    feasibility = round( # pyre-ignore
        rf_pred  * w.get("rf",0.33) +
        xgb_pred * w.get("xgb",0.34) +
        et_pred  * w.get("et",0.33), 1 # pyre-ignore
    )
    feasibility = max(0, min(100, feasibility))

    bmtpc_penalty = 0.0
    if combined.get("bmtpc_failure_count_25km", 0) > 0:
        severity = combined.get("bmtpc_failure_severity_index", 0) or 0
        bmtpc_penalty = min(10.0, max(5.0, severity * 2.0))
        feasibility = max(0, round(feasibility - bmtpc_penalty, 1)) # pyre-ignore

    # 9. Risk + Foundation
    risk = "Low Risk" if feasibility>=70 else "Medium Risk" if feasibility>=45 else "High Risk"

    lq = soil.get("liquefaction_risk", "Low")
    ss = soil.get("shrink_swell_risk", "Low")
    building = str(building_type)
    type_weight = {
        "House": 1.0,
        "Apartment": 1.1,
        "School": 1.2,
        "Hospital": 1.4,
        "Factory": 1.5,
        "Warehouse": 1.3,
        "Bridge": 1.6,
        "Mall": 1.4,
        "High Rise": 1.6,
    }
    load_weight = type_weight.get(building, 1.0)
    effective_bc = bc / load_weight
    if lq == "High":
        foundation = "Pile Foundation (Deep)"
    elif effective_bc < 60 or ss == "High":
        foundation = "Raft Foundation"
    elif building in {"Hospital", "Mall"}:
        foundation = "Isolated Footing with RCC"
    elif effective_bc >= 150:
        foundation = "Simple Strip Footing"
    else:
        foundation = "Isolated Footing"

    life_low  = max(10, int(life_pred - 10))
    life_high = int(life_pred + 10)
    conf      = round(50 + (feasibility/100)*40, 1) # pyre-ignore

    # 10. Requirement-specific scores
    soil_degradation_risk_score = round(max(0, min(100, 100 - soil_score)), 1) # pyre-ignore
    climate_stress_frequency_score = round(max(0, min(100, combined.get("climate_stress_index", 0))), 1)
    river_dist = combined.get("nearest_river_dist_km", 50)
    flood_risk = combined.get("flood_risk", "Low")
    water_exposure_probability_score = round(max(0, min(100, # pyre-ignore
        (combined.get("annual_rainfall_mm", 1000) / 3000) * 50 + # pyre-ignore
        max(0, (50 - min(100, river_dist))) * 0.5 +
        (20 if flood_risk == "High" else 10 if flood_risk == "Medium" else 0) # pyre-ignore
    )), 1)
    biological_damage_probability_score = round(max(0, min(100, combined.get("bio_pressure_index", 0))), 1)

    type_map = {
        "House": 1,
        "Apartment": 2,
        "School": 3,
        "Hospital": 4,
        "Factory": 5,
        "Warehouse": 6,
        "Bridge": 7,
        "Mall": 8,
    }
    construction_type_encoded = type_map.get(str(building_type), 0)

    # 11. Risk factor summary
    factors = []
    if soil_degradation_risk_score >= 60:
        factors.append("high soil degradation")
    if climate_stress_frequency_score >= 60:
        factors.append("high climate stress")
    if water_exposure_probability_score >= 60:
        factors.append("high water exposure")
    if biological_damage_probability_score >= 60:
        factors.append("high biological pressure")
    if combined.get("bmtpc_failure_count_25km", 0) > 0:
        factors.append("nearby BMTPC failure labels")
    if not factors:
        factors.append("overall stable conditions")
    risk_factor_summary = " + ".join(factors)

    safety_notes = _build_safety_notes(soil, climate, env, animal)

    # AHP-style comparison (simple weighted score for benchmarking)
    ahp_score = round( # pyre-ignore
        soil_score * 0.30 +
        climate_score * 0.25 +
        env_score * 0.30 +
        animal_score * 0.15,
        1 # pyre-ignore
    )
    ahp_delta = round(feasibility - ahp_score, 1)

    ok_count = sum(1 for k in ["bhuvan", "soilgrids", "cgwb", "nasa_power", "usgs", "gbif", "bmtpc_labels", "historical_dataset", "local_sensors"]
                   if source_status.get(k) == "ok")
    data_quality_score = round(ok_count / 9 * 100, 1) # pyre-ignore

    hist_updated = None
    hist_path = get_hist_path()
    if os.path.exists(hist_path):
        hist_updated = datetime.fromtimestamp(os.path.getmtime(hist_path), tz=timezone.utc).isoformat()

    return {
        "feasibility_score"  : feasibility,
        "risk_level"         : risk,
        "lifespan"           : f"{life_low}–{life_high} years",
        "confidence"         : conf,
        "foundation"         : foundation,
        "success_probability": round(succ_pred, 2), # pyre-ignore
        "bmtpc_failure_nearest_km"       : combined.get("bmtpc_failure_nearest_km"),
        "bmtpc_failure_count_25km"       : combined.get("bmtpc_failure_count_25km"),
        "bmtpc_failure_severity_index"   : combined.get("bmtpc_failure_severity_index"),
        "bmtpc_penalty"                  : round(bmtpc_penalty, 1), # pyre-ignore
        "soil_degradation_risk_score"      : soil_degradation_risk_score,
        "climate_stress_frequency_score"   : climate_stress_frequency_score,
        "water_exposure_probability_score" : water_exposure_probability_score,
        "biological_damage_probability_score": biological_damage_probability_score,
        "construction_type_encoded"        : construction_type_encoded,
        "risk_factor_summary"              : risk_factor_summary,
        "safety_notes"                     : safety_notes,
        "ahp_score"                        : ahp_score,
        "ahp_delta"                        : ahp_delta,
        "historical_match_km"              : round(hist_dist, 2) if hist_dist is not None else None, # pyre-ignore
        "data_quality_score"               : data_quality_score,
        "model_metadata": {
            "version": MODEL_VERSION,
            "trained_on": MODEL_TRAINED_ON,
            "feature_count": len(feature_list)
        },
        "data_freshness": {
            "bhuvan": fetched_at if BHUVAN_API_URL else None,
            "cgwb": fetched_at if CGWB_API_URL else None,
            "soilgrids": fetched_at,
            "nasa_power": fetched_at,
            "usgs": fetched_at,
            "gbif": fetched_at,
            "bmtpc_labels": datetime.fromtimestamp(os.path.getmtime(BMTPC_LABELS_PATH), tz=timezone.utc).isoformat() if os.path.exists(BMTPC_LABELS_PATH) else None,
            "historical_dataset": hist_updated,
            "local_sensors": sensor_data.get("sensor_timestamp") if sensor_data else None
        },
        "source_status": source_status,
        "domain_scores": {
            "soil"       : round(100 - soil_degradation_risk_score, 1),
            "climate"    : round(100 - climate_stress_frequency_score, 1),
            "environment": round(100 - water_exposure_probability_score, 1),
            "animal"     : round(100 - biological_damage_probability_score, 1)
        },
        "raw_data": {
            "soil": soil,
            "climate": climate,
            "env": env,
            "animal": animal,
            "bmtpc"  : bmtpc_risk,
        }
    }