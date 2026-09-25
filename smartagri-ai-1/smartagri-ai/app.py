"""
SmartAgri AI — Flask backend
=============================
A multilingual (English / Telugu / Tamil / Hindi) agriculture assistant.

Features:
- Dashboard with a farmer's field data (village, crop, soil moisture,
  temperature, humidity) stored in SQLite, plus a rule-based irrigation alert.
- Leaf disease "detection" endpoint. No trained model ships with this
  prototype (see model/README.md) so predictions are produced by a
  deterministic simulated classifier — swap in a real Keras/TFLite model
  by replacing `simulate_disease_prediction()`.
- Live weather via the free Open-Meteo API (geocoding + forecast, no API
  key required) with simple rule-based farming advice.
- A voice-assistant text endpoint that pairs with the browser's built-in
  Web Speech API (speech-to-text and text-to-speech run client-side).

Ready for Render: see Procfile (gunicorn) and requirements.txt.
"""

import os
import sqlite3
import hashlib
import random
from datetime import datetime

import requests
from flask import Flask, render_template, request, jsonify, g

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "smartagri.db")

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False  # keep non-Latin script readable in JSON responses


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS field_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            village TEXT NOT NULL,
            crop_name TEXT NOT NULL,
            soil_moisture REAL NOT NULL,
            temperature REAL NOT NULL,
            humidity REAL NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS disease_scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            disease_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Domain logic
# ---------------------------------------------------------------------------
def irrigation_alert(soil_moisture):
    """Simple rule-based irrigation guidance from soil moisture percentage."""
    if soil_moisture < 30:
        return "low", "low_moisture_alert", "Soil is dry — irrigate within 24 hours."
    if soil_moisture > 70:
        return "high", "high_moisture_alert", "Soil is very wet — hold off on irrigation."
    return "ok", "ok_moisture_alert", "Moisture is healthy — no irrigation needed today."


DISEASE_LIBRARY = [
    {
        "name": "Healthy leaf",
        "treatment": "No disease detected. Keep up regular watering and monitoring.",
    },
    {
        "name": "Early blight",
        "treatment": "Remove affected leaves, apply a copper-based fungicide, and avoid overhead watering.",
    },
    {
        "name": "Leaf rust",
        "treatment": "Apply a recommended fungicide and improve airflow between plants.",
    },
    {
        "name": "Bacterial leaf spot",
        "treatment": "Remove infected leaves, avoid working in wet fields, and rotate crops next season.",
    },
    {
        "name": "Powdery mildew",
        "treatment": "Apply a sulfur-based spray and reduce humidity around the plant where possible.",
    },
]


def simulate_disease_prediction(file_bytes):
    """
    Deterministic stand-in for a real CNN leaf-disease classifier.
    Hashing the uploaded image keeps the same photo giving the same
    result, so a demo doesn't look random on repeat uploads.
    Swap this out for real inference (e.g. TensorFlow/Keras on
    model/plant_model.h5) when a trained model is available.
    """
    digest = hashlib.sha256(file_bytes).hexdigest()
    seed = int(digest[:8], 16)
    rng = random.Random(seed)
    choice = rng.choice(DISEASE_LIBRARY)
    confidence = round(rng.uniform(78, 97), 1)
    return choice["name"], confidence, choice["treatment"]


def weather_advice(temp_c, rain_chance, wind_kmh, lang):
    """Very small rule-based advisory generator, per language."""
    msgs = {
        "en": [],
        "te": [],
        "ta": [],
        "hi": [],
    }
    if rain_chance >= 60:
        msgs["en"].append("High chance of rain — delay spraying and fertilizing today.")
        msgs["te"].append("వర్షం అవకాశం ఎక్కువ — ఈరోజు స్ప్రే మరియు ఎరువులు వేయడం వాయిదా వేయండి.")
        msgs["ta"].append("மழை வாய்ப்பு அதிகம் — இன்று தெளிப்பு மற்றும் உரமிடுதலை தள்ளி வையுங்கள்.")
        msgs["hi"].append("बारिश की अधिक संभावना — आज छिड़काव और खाद डालना टालें।")
    elif temp_c >= 35:
        msgs["en"].append("High heat expected — irrigate early morning or evening to reduce water loss.")
        msgs["te"].append("అధిక వేడి అంచనా — నీటి నష్టం తగ్గించడానికి ఉదయం లేదా సాయంత్రం నీరు పెట్టండి.")
        msgs["ta"].append("அதிக வெப்பம் எதிர்பார்க்கப்படுகிறது — நீர் இழப்பைக் குறைக்க காலை அல்லது மாலையில் பாசனம் செய்யுங்கள்.")
        msgs["hi"].append("अधिक गर्मी की संभावना — पानी की कमी रोकने के लिए सुबह या शाम को सिंचाई करें।")
    elif wind_kmh >= 25:
        msgs["en"].append("Windy conditions — avoid spraying pesticide, as drift will reduce accuracy.")
        msgs["te"].append("గాలి ఎక్కువగా ఉంది — పురుగుమందు స్ప్రే చేయడం మానుకోండి.")
        msgs["ta"].append("பலத்த காற்று — பூச்சிக்கொல்லி தெளிப்பதை தவிர்க்கவும்.")
        msgs["hi"].append("तेज़ हवा — कीटनाशक का छिड़काव करने से बचें।")
    else:
        msgs["en"].append("Conditions look favorable for regular field work today.")
        msgs["te"].append("ఈరోజు సాధారణ పొలం పనులకు పరిస్థితులు అనుకూలంగా ఉన్నాయి.")
        msgs["ta"].append("இன்று வழக்கமான வயல் வேலைகளுக்கு நிலைமைகள் சாதகமாக உள்ளன.")
        msgs["hi"].append("आज सामान्य खेत के काम के लिए स्थितियाँ अनुकूल हैं।")

    return msgs.get(lang, msgs["en"])[0]


CROP_REASONS = {
    "rice": {
        "en": "High rainfall and warm temperature favor a water-loving crop like rice.",
        "te": "ఎక్కువ వర్షపాతం మరియు వెచ్చని ఉష్ణోగ్రత వరి వంటి నీటిని ఇష్టపడే పంటకు అనుకూలం.",
        "ta": "அதிக மழைப்பொழிவு மற்றும் வெப்பமான வெப்பநிலை நெல் போன்ற நீரை விரும்பும் பயிருக்கு ஏற்றது.",
        "hi": "अधिक वर्षा और गर्म तापमान धान जैसी पानी पसंद करने वाली फ़सल के लिए अनुकूल है।",
    },
    "wheat": {
        "en": "Cooler temperature and moderate rainfall suit a winter cereal like wheat.",
        "te": "చల్లని ఉష్ణోగ్రత మరియు మోస్తరు వర్షపాతం గోధుమ వంటి శీతాకాల పంటకు అనుకూలం.",
        "ta": "குளிர்ந்த வெப்பநிலை மற்றும் மிதமான மழைப்பொழிவு கோதுமை போன்ற குளிர்கால தானியத்திற்கு ஏற்றது.",
        "hi": "ठंडा तापमान और मध्यम वर्षा गेहूं जैसी शीतकालीन फ़सल के लिए उपयुक्त है।",
    },
    "maize": {
        "en": "Balanced nutrients with moderate rainfall and warm temperature suit maize well.",
        "te": "మోస్తరు వర్షపాతం మరియు వెచ్చని ఉష్ణోగ్రతతో సమతుల్య పోషకాలు మొక్కజొన్నకు బాగా సరిపోతాయి.",
        "ta": "மிதமான மழைப்பொழிவு மற்றும் வெப்பமான வெப்பநிலையுடன் சமச்சீர் ஊட்டச்சத்துக்கள் சோளத்திற்கு ஏற்றது.",
        "hi": "मध्यम वर्षा और गर्म तापमान के साथ संतुलित पोषक तत्व मक्का के लिए उपयुक्त हैं।",
    },
    "cotton": {
        "en": "Low rainfall with high temperature and potassium levels favor cotton.",
        "te": "తక్కువ వర్షపాతం, అధిక ఉష్ణోగ్రత మరియు పొటాషియం స్థాయిలు పత్తికి అనుకూలం.",
        "ta": "குறைந்த மழைப்பொழிவு, அதிக வெப்பநிலை மற்றும் பொட்டாசியம் அளவுகள் பருத்திக்கு ஏற்றது.",
        "hi": "कम वर्षा, उच्च तापमान और पोटैशियम स्तर कपास के लिए अनुकूल हैं।",
    },
    "groundnut": {
        "en": "Well-drained soil with moderate rainfall and warm temperature suits groundnut.",
        "te": "మోస్తరు వర్షపాతం మరియు వెచ్చని ఉష్ణోగ్రతతో బాగా నీరు ఇంకే నేల వేరుశనగకు సరిపోతుంది.",
        "ta": "மிதமான மழைப்பொழிவு மற்றும் வெப்பமான வெப்பநிலையுடன் நன்கு வடிகட்டும் மண் நிலக்கடலைக்கு ஏற்றது.",
        "hi": "मध्यम वर्षा और गर्म तापमान के साथ अच्छी जल निकासी वाली मिट्टी मूंगफली के लिए उपयुक्त है।",
    },
    "pulses": {
        "en": "Lower nutrient demand and moderate conditions favor a nitrogen-fixing pulse crop.",
        "te": "తక్కువ పోషక అవసరం మరియు మోస్తరు పరిస్థితులు నత్రజనిని స్థిరీకరించే పప్పుధాన్యాల పంటకు అనుకూలం.",
        "ta": "குறைந்த ஊட்டச்சத்து தேவை மற்றும் மிதமான நிலைமைகள் நைட்ரஜனை நிலைநிறுத்தும் பருப்பு பயிருக்கு ஏற்றது.",
        "hi": "कम पोषक तत्व की मांग और मध्यम स्थितियां नाइट्रोजन स्थिर करने वाली दलहनी फ़सल के लिए अनुकूल हैं।",
    },
}

# order matters: first matching rule wins (simple decision-tree stand-in
# for a trained classifier — see README for wiring up a real model)
def simulate_crop_prediction(n, p, k, ph, temperature, rainfall, lang):
    if rainfall >= 180 and temperature >= 22:
        crop = "rice"
    elif temperature <= 20:
        crop = "wheat"
    elif rainfall < 90 and temperature >= 24:
        crop = "cotton"
    elif rainfall < 90 and temperature < 24:
        crop = "groundnut"
    elif k >= 40 and 90 <= rainfall < 180:
        crop = "maize"
    else:
        crop = "pulses"

    seed_str = f"{n}-{p}-{k}-{ph}-{temperature}-{rainfall}"
    digest = hashlib.sha256(seed_str.encode()).hexdigest()
    rng = random.Random(int(digest[:8], 16))
    confidence = round(rng.uniform(80, 96), 1)

    reason = CROP_REASONS.get(crop, CROP_REASONS["pulses"]).get(lang, CROP_REASONS[crop]["en"])
    crop_label = CROP_LABELS.get(crop, {}).get(lang, crop.capitalize())
    return crop_label, confidence, reason


CROP_LABELS = {
    "rice": {"en": "Rice", "te": "వరి", "ta": "நெல்", "hi": "धान"},
    "wheat": {"en": "Wheat", "te": "గోధుమ", "ta": "கோதுமை", "hi": "गेहूं"},
    "maize": {"en": "Maize", "te": "మొక్కజొన్న", "ta": "சோளம்", "hi": "मक्का"},
    "cotton": {"en": "Cotton", "te": "పత్తి", "ta": "பருத்தி", "hi": "कपास"},
    "sugarcane": {"en": "Sugarcane", "te": "చెరకు", "ta": "கரும்பு", "hi": "गन्ना"},
    "groundnut": {"en": "Groundnut", "te": "వేరుశనగ", "ta": "நிலக்கடலை", "hi": "मूंगफली"},
    "pulses": {"en": "Pulses", "te": "పప్పుధాన్యాలు", "ta": "பருப்பு வகைகள்", "hi": "दलहन"},
    "vegetables": {"en": "Vegetables", "te": "కూరగాయలు", "ta": "காய்கறிகள்", "hi": "सब्ज़ियां"},
}

# approximate kg-per-acre N-P-K guidance — a simplified prototype figure,
# not a substitute for a local soil-test-based recommendation
FERTILIZER_BASE = {
    "rice": (48, 24, 24),
    "wheat": (40, 20, 20),
    "maize": (50, 25, 20),
    "cotton": (40, 20, 20),
    "sugarcane": (80, 40, 40),
    "groundnut": (12, 24, 24),
    "pulses": (10, 20, 15),
    "vegetables": (35, 25, 25),
}

FERTILIZER_SENTENCE = {
    "en": "Apply approximately {n} kg Nitrogen, {p} kg Phosphorus, and {k} kg Potassium per acre for {crop}.",
    "te": "{crop} కోసం ఎకరానికి సుమారు {n} కిలోల నత్రజని, {p} కిలోల భాస్వరం, {k} కిలోల పొటాషియం వేయండి.",
    "ta": "{crop}க்கு ஏக்கருக்கு சுமார் {n} கிலோ நைட்ரஜன், {p} கிலோ பாஸ்பரஸ், {k} கிலோ பொட்டாசியம் இடவும்.",
    "hi": "{crop} के लिए प्रति एकड़ लगभग {n} किलो नाइट्रोजन, {p} किलो फॉस्फोरस, और {k} किलो पोटैशियम डालें।",
}

FERTILIZER_TIP = {
    "acidic": {
        "en": "Soil is acidic — consider adding agricultural lime before fertilizing.",
        "te": "నేల ఆమ్లంగా ఉంది — ఎరువు వేయడానికి ముందు వ్యవసాయ సున్నం జోడించడాన్ని పరిగణించండి.",
        "ta": "மண் அமிலத்தன்மை கொண்டது — உரமிடுவதற்கு முன் விவசாய சுண்ணாம்பு சேர்க்க பரிசீலிக்கவும்.",
        "hi": "मिट्टी अम्लीय है — खाद डालने से पहले कृषि चूना डालने पर विचार करें।",
    },
    "alkaline": {
        "en": "Soil is alkaline — add organic compost or gypsum to improve nutrient uptake.",
        "te": "నేల క్షారంగా ఉంది — పోషకాల శోషణను మెరుగుపరచడానికి సేంద్రియ ఎరువు లేదా జిప్సం జోడించండి.",
        "ta": "மண் காரத்தன்மை கொண்டது — ஊட்டச்சத்து உறிஞ்சலை மேம்படுத்த ஆர்கானிக் உரம் அல்லது ஜிப்சம் சேர்க்கவும்.",
        "hi": "मिट्टी क्षारीय है — पोषक तत्व अवशोषण सुधारने के लिए जैविक खाद या जिप्सम डालें।",
    },
    "neutral": {
        "en": "Soil pH is in a good range — split the fertilizer into 2-3 doses through the growing season.",
        "te": "నేల pH మంచి పరిధిలో ఉంది — పెరుగుదల కాలంలో ఎరువును 2-3 మోతాదులుగా విభజించండి.",
        "ta": "மண் pH நல்ல வரம்பில் உள்ளது — வளர்ச்சி காலத்தில் உரத்தை 2-3 அளவுகளாக பிரிக்கவும்.",
        "hi": "मिट्टी का pH अच्छी सीमा में है — बढ़ते मौसम में खाद को 2-3 खुराकों में बांटें।",
    },
}


def fertilizer_recommendation(crop, ph, lang):
    n, p, k = FERTILIZER_BASE.get(crop, FERTILIZER_BASE["pulses"])
    crop_label = CROP_LABELS.get(crop, {}).get(lang, crop.capitalize())
    sentence = FERTILIZER_SENTENCE.get(lang, FERTILIZER_SENTENCE["en"]).format(
        n=n, p=p, k=k, crop=crop_label
    )
    if ph < 5.5:
        tip_key = "acidic"
    elif ph > 7.5:
        tip_key = "alkaline"
    else:
        tip_key = "neutral"
    tip = FERTILIZER_TIP[tip_key].get(lang, FERTILIZER_TIP[tip_key]["en"])
    return sentence, tip


VOICE_ANSWERS = {
    # very small keyword-matched knowledge base; replace with a real
    # NLU/LLM call for production use.
    "en": {
        "soil": "Test your soil pH before sowing — most crops do best between 6.0 and 7.5.",
        "water": "Water early morning or late evening to reduce evaporation loss.",
        "fertilizer": "Use fertilizer based on a soil test — nitrogen for leafy growth, phosphorus for roots, potassium for fruiting.",
        "disease": "Open the Disease Scan page and upload a leaf photo for an instant check.",
        "crop": "Rice, maize, and pulses are commonly suited to loamy soil with moderate rainfall — check your local advisory for specifics.",
        "default": "Based on the sample soil and weather data, a moderately drought-tolerant crop with regular watering should do well. Try the Disease Scan or Weather pages for more specific help.",
    },
    "te": {
        "soil": "విత్తనాలు వేసే ముందు మీ నేల pH పరీక్షించండి — చాలా పంటలు 6.0 నుండి 7.5 మధ్య బాగా పెరుగుతాయి.",
        "water": "ఆవిరి నష్టం తగ్గించడానికి ఉదయం లేదా సాయంత్రం నీరు పెట్టండి.",
        "fertilizer": "నేల పరీక్ష ఆధారంగా ఎరువులు వాడండి — ఆకుల పెరుగుదలకు నత్రజని, వేర్లకు భాస్వరం, కాయలకు పొటాషియం.",
        "disease": "తక్షణ తనిఖీ కోసం వ్యాధి స్కాన్ పేజీని తెరచి ఆకు ఫోటో అప్‌లోడ్ చేయండి.",
        "crop": "మీ నేల మరియు వర్షపాతాన్ని బట్టి వరి, మొక్కజొన్న, పప్పుధాన్యాలు సాధారణంగా అనుకూలంగా ఉంటాయి.",
        "default": "నమూనా నేల మరియు వాతావరణ డేటా ఆధారంగా, క్రమం తప్పకుండా నీరు పెట్టే మోస్తరు కరువు-తట్టుకునే పంట బాగా పెరుగుతుంది.",
    },
    "ta": {
        "soil": "விதைப்பதற்கு முன் உங்கள் மண் pH ஐ சோதிக்கவும் — பெரும்பாலான பயிர்கள் 6.0 முதல் 7.5 வரை நன்றாக வளரும்.",
        "water": "ஆவியாதலைக் குறைக்க காலை அல்லது மாலை நேரத்தில் நீர் பாய்ச்சுங்கள்.",
        "fertilizer": "மண் பரிசோதனையின் அடிப்படையில் உரம் இடுங்கள் — இலை வளர்ச்சிக்கு நைட்ரஜன், வேர்களுக்கு பாஸ்பரஸ், காய்ப்புக்கு பொட்டாசியம்.",
        "disease": "உடனடி சோதனைக்கு நோய் ஸ்கேன் பக்கத்தைத் திறந்து இலை படத்தை பதிவேற்றவும்.",
        "crop": "உங்கள் மண் மற்றும் மழைப்பொழிவைப் பொறுத்து நெல், சோளம், பருப்பு வகைகள் பொதுவாக ஏற்றதாக இருக்கும்.",
        "default": "மாதிரி மண் மற்றும் வானிலை தரவின் அடிப்படையில், முறையான நீர்ப்பாசனத்துடன் மிதமான வறட்சி தாங்கும் பயிர் நன்றாக வளரும்.",
    },
    "hi": {
        "soil": "बुवाई से पहले अपनी मिट्टी का pH जांचें — अधिकांश फ़सलें 6.0 से 7.5 के बीच सबसे अच्छी होती हैं।",
        "water": "वाष्पीकरण कम करने के लिए सुबह जल्दी या शाम को पानी दें।",
        "fertilizer": "मिट्टी परीक्षण के आधार पर खाद डालें — पत्तों के लिए नाइट्रोजन, जड़ों के लिए फॉस्फोरस, फल के लिए पोटैशियम।",
        "disease": "तुरंत जांच के लिए रोग स्कैन पेज खोलें और पत्ती की फोटो अपलोड करें।",
        "crop": "आपकी मिट्टी और बारिश के अनुसार धान, मक्का और दलहन आमतौर पर उपयुक्त होते हैं।",
        "default": "नमूना मिट्टी और मौसम डेटा के आधार पर, नियमित सिंचाई के साथ मध्यम सूखा-सहनशील फ़सल अच्छी रहेगी।",
    },
}


def answer_voice_question(question, lang):
    q = (question or "").lower()
    bank = VOICE_ANSWERS.get(lang, VOICE_ANSWERS["en"])
    # naive keyword match across English trigger words regardless of UI language,
    # since speech-to-text may still transcribe some words in Latin script
    if any(w in q for w in ["soil", "మట్టి", "నేల", "மண்", "मिट्टी", "ph"]):
        return bank["soil"]
    if any(w in q for w in ["water", "irrigat", "నీరు", "நீர்", "पानी", "सिंचाई"]):
        return bank["water"]
    if any(w in q for w in ["fertil", "ఎరువు", "உரம்", "खाद", "उर्वरक"]):
        return bank["fertilizer"]
    if any(w in q for w in ["disease", "వ్యాధి", "நோய்", "रोग", "बीमारी", "pest"]):
        return bank["disease"]
    if any(w in q for w in ["crop", "పంట", "பயிர்", "फसल", "suitable"]):
        return bank["crop"]
    return bank["default"]


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    db = get_db()
    row = db.execute(
        "SELECT * FROM field_data ORDER BY id DESC LIMIT 1"
    ).fetchone()
    field = None
    if row:
        level, key, text = irrigation_alert(row["soil_moisture"])
        field = {
            "village": row["village"],
            "crop_name": row["crop_name"],
            "soil_moisture": row["soil_moisture"],
            "temperature": row["temperature"],
            "humidity": row["humidity"],
            "alert_level": level,
            "alert_key": key,
            "alert_text": text,
        }
    return render_template("dashboard.html", active="dashboard", field=field)


@app.route("/crop")
def crop():
    return render_template("crop.html", active="crop")


@app.route("/disease")
def disease():
    return render_template("disease.html", active="disease")


@app.route("/fertilizer")
def fertilizer():
    return render_template("fertilizer.html", active="fertilizer")


@app.route("/weather")
def weather():
    db = get_db()
    row = db.execute(
        "SELECT * FROM field_data ORDER BY id DESC LIMIT 1"
    ).fetchone()
    field = {"village": row["village"]} if row else None
    return render_template("weather.html", active="weather", field=field)


@app.route("/voice")
def voice():
    return render_template("voice.html", active="voice")


# ---------------------------------------------------------------------------
# REST API
# ---------------------------------------------------------------------------
@app.route("/api/field", methods=["GET", "POST"])
def api_field():
    db = get_db()
    if request.method == "POST":
        data = request.get_json(force=True) or {}
        village = str(data.get("village", "")).strip()
        crop_name = str(data.get("crop_name", "")).strip()
        try:
            soil_moisture = float(data.get("soil_moisture", 0))
            temperature = float(data.get("temperature", 0))
            humidity = float(data.get("humidity", 0))
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "Invalid numeric field"}), 400

        if not village or not crop_name:
            return jsonify({"ok": False, "error": "Village and crop name are required"}), 400

        db.execute(
            """INSERT INTO field_data
               (village, crop_name, soil_moisture, temperature, humidity, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (village, crop_name, soil_moisture, temperature, humidity, datetime.utcnow().isoformat()),
        )
        db.commit()

        level, key, text = irrigation_alert(soil_moisture)
        return jsonify({
            "ok": True,
            "field": {
                "village": village,
                "crop_name": crop_name,
                "soil_moisture": soil_moisture,
                "temperature": temperature,
                "humidity": humidity,
                "alert_level": level,
                "alert_key": key,
                "alert_text": text,
            },
        })

    row = db.execute("SELECT * FROM field_data ORDER BY id DESC LIMIT 1").fetchone()
    if not row:
        return jsonify({"ok": True, "field": None})
    level, key, text = irrigation_alert(row["soil_moisture"])
    return jsonify({
        "ok": True,
        "field": {
            "village": row["village"],
            "crop_name": row["crop_name"],
            "soil_moisture": row["soil_moisture"],
            "temperature": row["temperature"],
            "humidity": row["humidity"],
            "alert_level": level,
            "alert_key": key,
            "alert_text": text,
        },
    })


@app.route("/api/predict-disease", methods=["POST"])
def api_predict_disease():
    if "leaf" not in request.files:
        return jsonify({"error": "No leaf image uploaded"}), 400

    file = request.files["leaf"]
    file_bytes = file.read()

    if not file_bytes:
        return jsonify({"error": "Empty file"}), 400

    disease_name, confidence, treatment = simulate_disease_prediction(file_bytes)

    return jsonify({
        "disease_name": disease_name,
        "confidence": confidence,
        "treatment": treatment
    })


@app.route("/api/predict-crop", methods=["POST"])
def api_predict_crop():
    data = request.get_json(force=True) or {}
    try:
        n = float(data.get("n", 0))
        p = float(data.get("p", 0))
        k = float(data.get("k", 0))
        ph = float(data.get("ph", 6.5))
        temperature = float(data.get("temperature", 25))
        rainfall = float(data.get("rainfall", 100))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid numeric input"}), 400
    lang = data.get("lang", "en")

    crop_label, confidence, reason = simulate_crop_prediction(n, p, k, ph, temperature, rainfall, lang)
    return jsonify({"crop": crop_label, "confidence": confidence, "reason": reason})


@app.route("/api/fertilizer", methods=["POST"])
def api_fertilizer():
    data = request.get_json(force=True) or {}
    crop_key = str(data.get("crop", "rice")).lower()
    try:
        ph = float(data.get("ph", 6.5))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid soil pH"}), 400
    lang = data.get("lang", "en")

    if crop_key not in FERTILIZER_BASE:
        crop_key = "rice"

    sentence, tip = fertilizer_recommendation(crop_key, ph, lang)
    return jsonify({"fertilizer": sentence, "tip": tip})


@app.route("/api/weather")
def api_weather():
    village = request.args.get("village", "").strip()
    lang = request.args.get("lang", "en")
    if not village:
        return jsonify({"error": "Please provide a village or city name"}), 400

    try:
        geo_res = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": village, "count": 1, "language": "en", "format": "json"},
            timeout=8,
        )
        geo_data = geo_res.json()
        results = geo_data.get("results")
        if not results:
            return jsonify({"error": f"Could not find a location named '{village}'"}), 404

        place = results[0]
        lat, lon = place["latitude"], place["longitude"]
        place_label = ", ".join(
            filter(None, [place.get("name"), place.get("admin1"), place.get("country")])
        )

        wx_res = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation_probability",
                "hourly": "precipitation_probability",
                "timezone": "auto",
            },
            timeout=8,
        )
        wx_data = wx_res.json()
        current = wx_data.get("current", {})

        temperature = current.get("temperature_2m", 28)
        humidity = current.get("relative_humidity_2m", 55)
        wind_speed = current.get("wind_speed_10m", 8)
        rain_chance = current.get("precipitation_probability")
        if rain_chance is None:
            hourly_probs = wx_data.get("hourly", {}).get("precipitation_probability", [])
            rain_chance = hourly_probs[0] if hourly_probs else 10

        advice = weather_advice(temperature, rain_chance, wind_speed, lang)

        condition_map = {
            "en": "Live conditions", "te": "ప్రత్యక్ష పరిస్థితులు",
            "ta": "நேரடி நிலைமைகள்", "hi": "लाइव स्थिति",
        }

        return jsonify({
            "place": place_label,
            "temperature": temperature,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "rain_chance": rain_chance,
            "condition": condition_map.get(lang, condition_map["en"]),
            "advice": advice,
        })
    except requests.RequestException:
        return jsonify({"error": "Weather service is unavailable right now. Please try again."}), 502

VOICE_KB = {
    "en": {
        "irrigation": "Water your crop early morning or evening. Avoid midday irrigation to reduce evaporation loss.",
        "fertilizer": "Use a balanced N-P-K fertilizer based on your soil test. Apply it in split doses for better uptake.",
        "disease": "Check the leaves for spots or discoloration. Use the Disease Scan tab to upload a photo for analysis.",
        "crop": "Based on your soil and rainfall values, rice or maize is generally a good choice for this season.",
        "weather": "Check the Weather tab for live conditions and farming advice for today.",
        "default": "I can help with irrigation, fertilizer, crop selection, and disease questions. Could you ask about one of those?"
    },
    "te": {
        "irrigation": "మీ పంటకు ఉదయం లేదా సాయంత్రం నీరు పెట్టండి. మధ్యాహ్నం నీరు పెట్టడం వల్ల ఆవిరి నష్టం ఎక్కువ.",
        "fertilizer": "మీ నేల పరీక్ష ఆధారంగా సమతుల్య N-P-K ఎరువు వాడండి. మెరుగైన శోషణ కోసం విడతలుగా వేయండి.",
        "disease": "ఆకులపై మచ్చలు లేదా రంగు మార్పు చూడండి. విశ్లేషణ కోసం Disease Scan టాబ్‌లో ఫోటో అప్‌లోడ్ చేయండి.",
        "crop": "మీ నేల మరియు వర్షపాతం ఆధారంగా, ఈ సీజన్‌కు వరి లేదా మొక్కజొన్న మంచి ఎంపిక.",
        "weather": "ఈరోజు వాతావరణం మరియు వ్యవసాయ సలహా కోసం Weather టాబ్ చూడండి.",
        "default": "నేను నీటిపారుదల, ఎరువులు, పంట ఎంపిక మరియు వ్యాధుల గురించి సహాయం చేయగలను."
    },
    "ta": {
        "irrigation": "உங்கள் பயிருக்கு காலை அல்லது மாலை நேரத்தில் நீர் பாய்ச்சவும். மதியம் நீர்ப்பாசனம் தவிர்க்கவும்.",
        "fertilizer": "மண் பரிசோதனையின் அடிப்படையில் சமச்சீர் N-P-K உரத்தைப் பயன்படுத்தவும். பிரிவுகளாக இடவும்.",
        "disease": "இலைகளில் புள்ளிகள் அல்லது நிற மாற்றத்தை சரிபார்க்கவும். Disease Scan தாவலில் புகைப்படத்தை பதிவேற்றவும்.",
        "crop": "உங்கள் மண் மற்றும் மழைப்பொழிவின் அடிப்படையில், இந்த பருவத்திற்கு நெல் அல்லது சோளம் நல்ல தேர்வு.",
        "weather": "இன்றைய வானிலை மற்றும் விவசாய ஆலோசனைக்கு Weather தாவலைப் பார்க்கவும்.",
        "default": "நான் நீர்ப்பாசனம், உரம், பயிர் தேர்வு மற்றும் நோய்கள் குறித்து உதவ முடியும்."
    },
    "hi": {
        "irrigation": "अपनी फसल को सुबह या शाम को पानी दें। दोपहर में सिंचाई से बचें, इससे वाष्पीकरण ज़्यादा होता है।",
        "fertilizer": "अपनी मिट्टी की जांच के आधार पर संतुलित N-P-K उर्वरक का उपयोग करें। इसे भागों में डालें।",
        "disease": "पत्तियों पर धब्बे या रंग बदलाव देखें। विश्लेषण के लिए Disease Scan टैब में फोटो अपलोड करें।",
        "crop": "आपकी मिट्टी और वर्षा के आधार पर, इस मौसम के लिए चावल या मक्का एक अच्छा विकल्प है।",
        "weather": "आज के हालात और खेती सलाह के लिए Weather टैब देखें।",
        "default": "मैं सिंचाई, उर्वरक, फसल चयन और रोगों के बारे में मदद कर सकता हूँ।"
    },
}

def answer_voice_question(question, lang):
    q = (question or "").lower()
    kb = VOICE_KB.get(lang, VOICE_KB["en"])

    keywords = {
        "irrigation": ["water", "irrigat", "నీరు", "நீர்", "पानी", "सिंचाई", "நீர்ப்பாசனம்"],
        "fertilizer": ["fertiliz", "urea", "nutrient", "ఎరువు", "உரம்", "उर्वरक", "खाद"],
        "disease": ["disease", "leaf", "spot", "sick", "వ్యాధి", "நோய்", "रोग", "बीमारी"],
        "crop": ["crop", "grow", "plant", "suitable", "పంట", "பயிர்", "फसल", "उगा"],
        "weather": ["weather", "rain", "వాతావరణం", "வானிலை", "मौसम", "बारिश"],
    }

    for intent, words in keywords.items():
        if any(w in q for w in words):
            return kb[intent]

    return kb["default"]
@app.route("/api/voice-query", methods=["POST"])
def api_voice_query():
    data = request.get_json(force=True) or {}
    question = data.get("question", "")
    lang = data.get("lang", "en")
    answer = answer_voice_question(question, lang)
    return jsonify({"answer": answer})


# ---------------------------------------------------------------------------
if not os.path.exists(DB_PATH):
    init_db()
else:
    # make sure tables exist even if an empty db file was committed
    init_db()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
