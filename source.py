import streamlit as st
import json
import os
import requests
import pandas as pd
import base64
from PIL import Image
import io
import cv2
import numpy as np
from serpapi import GoogleSearch
from agno.agent import Agent
from agno.tools.serpapi import SerpApiTools
from agno.models.google import Gemini
from agno.exceptions import ModelProviderError
from twilio.rest import Client
from datetime import datetime
import pytesseract
import re             

import json
from supabase import create_client
import streamlit as st
import os
from supabase_utils import log_search, log_contact, get_analytics

SUPABASE_URL = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL"))
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_KEY"))
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

import streamlit as st
import os
from supabase import create_client
from supabase_utils import log_search, log_contact

# === Initialize Supabase ===
SUPABASE_URL = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL"))
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_KEY"))
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# === Session ID (optional for unique tracking) ===
import uuid
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# === Search Logging ===
def save_search(destination, budget_preference, travel_class, travel_days, user_id="guest"):
    try:
        log_search(destination, budget_preference, travel_class, travel_days, user_id)
    except Exception:
        pass  # Do not display or print anything

# === Contact Logging ===
def save_contact(name, email, message):
    try:
        log_contact(name, email, message)
    except Exception:
        pass  # Silent fail-safe

st.markdown("""
    <style>
        .title-container {
            text-align: center;
            margin-bottom: 20px;
        }
        .main-title {
            font-size: 48px;
            font-weight: bold;
            color: #4A4A6A;
            margin-bottom: 5px;
        }
        .subtitle {
            font-size: 24px;
            color: #777;
            margin-top: 0;
        }
        .stSlider > div { background-color: #f9f9f9; padding: 10px; border-radius: 10px; }
        .passport-scan { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 15px; margin: 20px 0; }
        .visa-free-card {
            background: #f8f9fa;
            border-left: 4px solid #28a745;
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .country-name {
            font-size: 16px;
            font-weight: bold;
            color: #333;
            margin-bottom: 5px;
        }
        .visa-status {
            font-size: 12px;
            color: #28a745;
            font-weight: 500;
        }
        .top-navigation {
            display: flex;
            justify-content: center;
            margin-bottom: 30px;
            gap: 20px;
            border-bottom: 1px solid #ddd;
            padding-bottom: 10px;
        }
        .top-navigation button {
            background-color: transparent;
            color: #667eea;
            border: none;
            padding: 10px 15px;
            font-size: 18px;
            cursor: pointer;
            transition: color 0.3s, border-bottom 0.3s;
        }
        .top-navigation button:hover {
            color: #764ba2;
            border-bottom: 2px solid #764ba2;
        }
        .top-navigation button.active {
            color: #764ba2;
            font-weight: bold;
            border-bottom: 2px solid #764ba2;
        }
        .flight-card {
            border: 2px solid #ddd;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
            background-color: #f9f9f9;
            margin-bottom: 20px;
        }
        .flight-card img {
            max-width: 100px;
            margin-bottom: 10px;
        }
        .flight-card h3 {
            margin: 10px 0;
        }
        .flight-card p {
            margin: 5px 0;
        }
        .flight-card .price {
            color: #008000;
            font-size: 24px;
            font-weight: bold;
            margin-top: 10px;
        }
        .flight-card .book-now-link {
            display: inline-block;
            padding: 10px 20px;
            font-size: 16px;
            font-weight: bold;
            color: #fff;
            background-color: #007bff;
            text-decoration: none;
            border-radius: 5px;
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        st.error(f"Logo file not found: {image_path}. Please add Roamlogo.png")
        return None

logo_path = "Roamlogo.png"
logo_base64 = get_base64_image(logo_path)

if 'passport_country' not in st.session_state:
    st.session_state.passport_country = None
if 'visa_free_countries' not in st.session_state:
    st.session_state.visa_free_countries = []
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Travel Plan"

if logo_base64:
    st.markdown(f"""
        <div class="title-container">
            <img src="data:image/png;base64,{logo_base64}" width="150" style="margin-bottom: 10px;">
            <div class="main-title">RoamGenie</div>
            <p class="subtitle">AI-Powered Travel Planner</p>
        </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
        <div class="title-container">
            <div class="main-title">RoamGenie</div>
            <p class="subtitle">AI-Powered Travel Planner</p>
        </div>
    """, unsafe_allow_html=True)


# ---== API KEYS ==---
# !! IMPORTANT: PASTE YOUR KEYS HERE !!
GOOGLE_API_KEY = "AIzaSyDcIuEAYI3nauiaPWxwbhxKp3U5IqWEr-I"
RAPIDAPI_KEY = "fc5b326458mshf73f3c8fca1d6d7p126a20jsn6fe65f2abcaa" 
SERPAPI_KEY = "c43e77359a094203b64a96124b6e7b3814d510ce7cef1fe11199ed03546cdcb5"
# ---================---

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

OCR_SPACE_API_KEY = "YOUR_OCR_SPACE_API_KEY"
MINDEE_API_KEY = "YOUR_MINDEE_API_KEY"

class PassportScanner:
    def __init__(self):
        self.visa_data = None
        self.country_flags = {}
        self.load_visa_dataset()
        self.load_country_flags()

    def load_visa_dataset(self):
        try:
            url1 = "https://raw.githubusercontent.com/ilyankou/passport-index-dataset/master/passport-index-tidy.csv"
            try:
                self.visa_data = pd.read_csv(url1)
                return
            except Exception as e1:
                st.warning(f"Failed to load primary dataset: {e1}. Attempting secondary...")

            url2 = "https_raw_githubusercontent_com_datasets_passport_index_main_data_passport_index_tidy.csv"
            try:
                self.visa_data = pd.read_csv(url2)
                return
            except Exception as e2:
                st.warning(f"Failed to load secondary dataset: {e2}. Creating comprehensive dataset as fallback...")

            self.create_comprehensive_visa_data()

        except Exception as e:
            st.error(f"Error loading visa dataset: {e}")
            self.create_comprehensive_visa_data()

    def create_comprehensive_visa_data(self):
        # ... (Your comprehensive visa data remains unchanged) ...
        visa_data = {
            'India': {
                'visa_free': [
                    'Bhutan', 'Nepal', 'Maldives', 'Mauritius', 'Seychelles', 'Fiji',
                    'Vanuatu', 'Micronesia', 'Samoa', 'Cook Islands', 'Niue', 'Tuvalu',
                    'Indonesia', 'Thailand', 'Malaysia', 'Singapore', 'Philippines',
                    'Cambodia', 'Laos', 'Myanmar', 'Sri Lanka', 'Bangladesh',
                    'South Korea', 'Japan', 'Qatar', 'UAE', 'Oman', 'Kuwait',
                    'Bahrain', 'Jordan', 'Iran', 'Armenia', 'Georgia', 'Kazakhstan',
                    'Kyrgyzstan', 'Tajikistan', 'Uzbekistan', 'Mongolia', 'Turkey',
                    'Serbia', 'Albania', 'North Macedonia', 'Bosnia and Herzegovina',
                    'Montenegro', 'Moldova', 'Belarus', 'Madagascar', 'Comoros',
                    'Cape Verde', 'Guinea-Bissau', 'Mozambique', 'Zimbabwe', 'Zambia',
                    'Uganda', 'Rwanda', 'Burundi', 'Tanzania', 'Kenya', 'Ethiopia',
                    'Djibouti', 'Somalia', 'Sudan', 'Egypt', 'Morocco', 'Tunisia',
                    'Barbados', 'Dominica', 'Grenada', 'Haiti', 'Jamaica',
                    'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Vincent and the Grenadines',
                    'Trinidad and Tobago', 'El Salvador', 'Honduras', 'Nicaragua',
                    'Bolivia', 'Ecuador', 'Suriname'
                ]
            },
            # ... other countries
        }

        passport_data = []
        destination_data = []
        requirement_data = []

        for passport_country, data in visa_data.items():
            for dest_country in data['visa_free']:
                passport_data.append(passport_country)
                destination_data.append(dest_country)
                requirement_data.append('visa free')

        self.visa_data = pd.DataFrame({
            'Passport': passport_data,
            'Destination': destination_data,
            'Requirement': requirement_data
        })


    def load_country_flags(self):
        # ... (Your country flags data remains unchanged) ...
        self.country_flags = { 'Thailand': '', 'Singapore': '', 'Malaysia': '', }


    def extract_passport_info_tesseract(self, image_file):
        try:
            image = Image.open(image_file)
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text = pytesseract.image_to_string(thresh, config='--psm 6')
            return self.parse_passport_text(text)

        except Exception as e:
            st.error(f"Tesseract OCR Error: {e}")
            return None

    def parse_passport_text(self, text):
        # ... (Your passport parsing logic remains unchanged) ...
        text = text.upper()
        country_patterns = [r'REPUBLIC OF ([A-Z\s]+)',]
        country_mapping = {'INDIA': 'India',}
        for pattern in country_patterns:
            match = re.search(pattern, text)
            if match:
                country = match.group(1) if match.groups() else match.group(0)
                country = country.strip()
                if country in country_mapping:
                    return {'country': country_mapping[country], 'confidence': 0.8}
        return None


    def get_visa_free_countries(self, passport_country):
        # ... (Your visa-free logic remains unchanged) ...
        if self.visa_data is None:
            st.error("Visa dataset not loaded")
            return []
        try:
            passport_country_clean = passport_country.strip()
            visa_free_data = self.visa_data[
                (self.visa_data['Passport'].str.strip().str.lower() == passport_country_clean.lower()) &
                (self.visa_data['Requirement'].str.contains('visa free|visa-free|visa on arrival', case=False, na=False))
            ]
            countries = sorted(visa_free_data['Destination'].unique().tolist())
            countries = [country for country in countries if country and str(country).strip()]
            st.success(f"Found {len(countries)} visa-free destinations for {passport_country_clean}")
            return countries
        except Exception as e:
            st.error(f"Error fetching visa-free countries: {e}")
            return []

passport_scanner = PassportScanner()

# Navigation Bar
col_nav = st.columns(5)
if col_nav[0].button("Travel Plan", key="nav_travel_plan", help="Plan your next trip"):
    st.session_state.current_page = "Travel Plan"
if col_nav[1].button("Passport", key="nav_passport", help="Find visa-free destinations based on your passport"):
    st.session_state.current_page = "Passport"
if col_nav[3].button("Contact Us", key="nav_contact_us", help="Get in touch with us"):
    st.session_state.current_page = "Contact Us"



st.markdown("---")

# Check for API keys
if SERPAPI_KEY == "YOUR_SERPAPI_KEY" or GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY" or RAPIDAPI_KEY == "YOUR_RAPIDAPI_KEY":
    st.error("Missing API Keys! Please paste your API keys at the top of the script (around line 600) to use the app.")
    st.stop()



# main.py — Unified RoamGenie backend (cleaned)
import os
import io
import re
import json
from datetime import datetime, date
from typing import Optional, List, Dict, Any

import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Optional image / OCR libs
try:
    import cv2
    import numpy as np
    from PIL import Image
    import pytesseract
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# Supabase Python client
try:
    from supabase import create_client, Client
except Exception:
    create_client = None
    Client = None

# Gemini (Google Generative) SDK (optional)
try:
    from google import generativeai as genai
except Exception:
    genai = None

# Load env
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")

SERPAPI_KEY = os.getenv("SERPAPI_KEY")        # optional - SerpAPI for flights
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")      # optional - RapidAPI for trains
GEMINI_ENABLED = os.getenv("GEMINI_ENABLED", "false").lower() == "true"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # optional - if using Google Generative API

# Initialize Gemini if requested and library available
GEMINI_MODEL = None
if GEMINI_ENABLED and genai and GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        GEMINI_MODEL = genai.GenerativeModel("gemini-2.0-flash")
    except Exception as e:
        print("⚠️ Gemini init error:", e)
        GEMINI_MODEL = None
        GEMINI_ENABLED = False

# Initialize Supabase client (if credentials present and library available)
supabase = None
if create_client and SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("🔑 Supabase client initialized")
    except Exception as e:
        print("⚠️ Supabase init error:", e)
        supabase = None
else:
    supabase = None

# FastAPI app
app = FastAPI(title="RoamGenie API (Unified)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN, "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# MODELS
# -------------------------
class SearchPayload(BaseModel):
    user_id: Optional[str] = "guest"
    departure_city: Optional[str] = None
    destination: str
    days: Optional[int] = None
    theme: Optional[str] = None
    activities: Optional[str] = None
    budget_preference: Optional[str] = None
    travel_class: Optional[str] = None
    departure_date: Optional[str] = None
    return_date: Optional[str] = None

class GeneratePayload(SearchPayload):
    # stricter for some endpoints, but keep compatible
    pass

class ModifyPayload(BaseModel):
    user_id: Optional[str] = "guest"
    current_itinerary: str
    modification_prompt: str
    context: Optional[dict] = None

class PackagePayload(BaseModel):
    user_id: Optional[str] = "guest"
    itinerary: str
    context: dict

class ContactPayload(BaseModel):
    first_name: str
    last_name: str
    email: str
    phone: Optional[str] = None
    message: Optional[str] = ""
    source: Optional[str] = "RoamGenie Website"

# -------------------------
# Utilities
# -------------------------
def format_datetime(iso_string: Optional[str]):
    if not iso_string:
        return "N/A"
    try:
        if "T" in iso_string:
            dt = datetime.fromisoformat(iso_string)
            return dt.strftime("%b-%d, %Y | %I:%M %p")
        dt = datetime.strptime(iso_string, "%Y-%m-%d")
        return dt.strftime("%b-%d, %Y")
    except Exception:
        return iso_string

# -------------------------
# Flights / Trains helpers
# -------------------------
def fetch_flights_serpapi(departure_id: str, arrival_id: str, outbound_date: str, return_date: Optional[str]):
    if not SERPAPI_KEY:
        return {}
    params = {
        "engine": "google_flights",
        "departure_id": departure_id,
        "arrival_id": arrival_id,
        "outbound_date": outbound_date,
        "currency": "INR",
        "hl": "en",
        "api_key": SERPAPI_KEY,
    }
    if return_date:
        params["return_date"] = return_date
    try:
        resp = requests.get("https://serpapi.com/search", params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print("⚠️ SerpAPI flights error:", e)
        return {}

def normalize_flights_from_serp(api_json: dict):
    best_flights = api_json.get("best_flights", []) if isinstance(api_json, dict) else []
    normalized = []
    if not best_flights:
        return normalized
    sorted_f = sorted(best_flights, key=lambda x: x.get("price", float("inf")))[:3]
    for f in sorted_f:
        flights_details = f.get("flights", [{}])
        dep = flights_details[0].get("departure_airport", {}) if flights_details else {}
        arr = flights_details[-1].get("arrival_airport", {}) if flights_details else {}
        provider = f.get("airline") or "Unknown Airline"
        logo = f.get("airline_logo") or ""
        price = f.get("price")
        booking_link = f.get("link") or f"https://www.google.com/flights?q={dep.get('id','')}{arr.get('id','')}"
        normalized.append({
            "type": "Flight",
            "provider": provider,
            "provider_logo": logo,
            "price": price,
            "departure_time": format_datetime(dep.get("time", "N/A")),
            "arrival_time": format_datetime(arr.get("time", "N/A")),
            "duration_minutes": f.get("total_duration", "N/A"),
            "booking_link": booking_link
        })
    return normalized

def fetch_trains_rapidapi(from_station: str, to_station: str, date_str: str):
    if not RAPIDAPI_KEY:
        return {}
    url = "https://irctc1.p.rapidapi.com/api/v3/trainBetweenStations"
    qs = {"fromStationCode": from_station, "toStationCode": to_station, "dateOfJourney": date_str}
    headers = {"x-rapidapi-key": RAPIDAPI_KEY, "x-rapidapi-host": "irctc1.p.rapidapi.com"}
    try:
        resp = requests.get(url, headers=headers, params=qs, timeout=12)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print("⚠️ RapidAPI trains error:", e)
        return {}

def normalize_trains_from_rapid(api_json, departure_date_obj: date):
    normalized = []
    if not api_json or not isinstance(api_json, dict):
        return normalized
    train_list = api_json.get("data") or api_json.get("trains") or []
    for train in train_list[:5]:
        try:
            train_name = train.get("train_name", "Unknown Train")
            train_number = train.get("train_number", "")
            from_code = train.get("from", "")
            to_code = train.get("to", "")
            from_std = train.get("from_std", "") or train.get("departure_time", "")
            to_sta = train.get("to_sta", "") or train.get("arrival_time", "")
            duration = train.get("duration", "")
            date_str = departure_date_obj.strftime("%d-%m-%Y")
            booking_link = f"https://www.irctc.co.in/nget/train-search?fromStation={from_code}&toStation={to_code}&journeyDate={date_str}"
            normalized.append({
                "type": "Train",
                "provider": f"{train_name} ({train_number})",
                "provider_logo": "",
                "price": None,
                "departure_time": from_std,
                "arrival_time": to_sta,
                "duration_minutes": duration,
                "booking_link": booking_link
            })
        except Exception:
            continue
    return normalized

# -------------------------
# Fallback itinerary & packages (single canonical implementation)
# -------------------------
def fallback_create_itinerary(payload: GeneratePayload, flights: List[dict], trains: List[dict], hotels_txt: str) -> str:
    days = max(1, int(payload.days or 1))
    lines = []
    lines.append(f"**{payload.destination} {days}-Day Itinerary**")
    lines.append("")
    for d in range(1, days + 1):
        lines.append(f"**Day {d}:**")
        lines.append(f"- Morning: Explore landmarks and local neighborhoods.")
        lines.append(f"- Afternoon: Try local cuisine and visit a museum or market.")
        lines.append(f"- Evening: Relax and enjoy local entertainment.")
        lines.append("")
    lines.append(f"**Trip Theme:** {payload.theme or 'General'}")
    if payload.departure_date and payload.return_date:
        lines.append(f"**Travel Dates:** {payload.departure_date} → {payload.return_date}")
    lines.append("")
    if flights:
        lines.append("**Suggested Flights (top 3):**")
        for f in flights:
            lines.append(f"- {f.get('provider')} | {f.get('departure_time')} → {f.get('arrival_time')} | ₹{f.get('price', 'N/A')}")
    if trains:
        lines.append("**Suggested Trains (top picks):**")
        for t in trains:
            lines.append(f"- {t.get('provider')} | {t.get('departure_time')} → {t.get('arrival_time')}")
    if hotels_txt:
        lines.append("")
        lines.append("**Hotels & Restaurants (summary):**")
        lines.append(hotels_txt[:1000])
    return "\n".join(lines)

def fallback_generate_packages(itinerary_text: str, context: dict, cheapest_flights: list) -> List[dict]:
    # basic deterministic rule-based packages (INR)
    flight_cost = 0
    if cheapest_flights and len(cheapest_flights) > 0:
        try:
            flight_cost = int(cheapest_flights[0].get("price") or 0)
        except Exception:
            flight_cost = 0

    days = context.get("num_days") or context.get("days") or 1
    days = max(1, int(days))
    packages = []
    packages.append({
        "package_name": "Budget",
        "estimated_hotel_cost": 3000 * days,
        "estimated_food_cost": 1000 * days,
        "estimated_transport_cost": 500 * days,
        "activity_cost": 1500,
        "description": "This package uses budget-friendly hotels and focuses on local eateries and public transport.",
        "flight_cost": flight_cost,
    })
    packages.append({
        "package_name": "Standard",
        "estimated_hotel_cost": 8000 * days,
        "estimated_food_cost": 2000 * days,
        "estimated_transport_cost": 1500 * days,
        "activity_cost": 2500,
        "description": "A balanced plan with comfortable hotels and occasional private transport.",
        "flight_cost": flight_cost,
    })
    packages.append({
        "package_name": "Luxury",
        "estimated_hotel_cost": 20000 * days,
        "estimated_food_cost": 5000 * days,
        "estimated_transport_cost": 4000 * days,
        "activity_cost": 5000,
        "description": "Premium hotels, fine dining, and private transport.",
        "flight_cost": flight_cost,
    })
    for p in packages:
        total = int(p["estimated_hotel_cost"]) + int(p["estimated_food_cost"]) + int(p["estimated_transport_cost"]) + int(p["activity_cost"]) + int(p.get("flight_cost", 0))
        p["total_cost"] = total
    return packages

# -------------------------
# Gemini wrappers (best-effort, parse JSON safely)
# -------------------------
def call_gemini_text(prompt: str, max_output_tokens: int = 800) -> Optional[str]:
    if not GEMINI_ENABLED or not GEMINI_MODEL:
        return None
    try:
        resp = GEMINI_MODEL.generate_content(prompt)
        if hasattr(resp, "text"):
            return resp.text
        if isinstance(resp, dict):
            return resp.get("text") or resp.get("output") or json.dumps(resp)
        return str(resp)
    except Exception as e:
        print("❌ Gemini call error:", e)
        return None

def call_gemini_packages(itinerary: str, context: dict, cheapest_flights: list) -> Optional[List[dict]]:
    if not GEMINI_ENABLED or not GEMINI_MODEL:
        return None
    prompt = f"""
You are a travel packaging expert. Given this itinerary and context, produce 3 package tiers (Budget, Standard, Luxury) as a JSON array.
Each package must include numeric cost fields: package_name, estimated_hotel_cost, estimated_food_cost, estimated_transport_cost, activity_cost, flight_cost, description, total_cost.
Return only valid JSON (an array of objects). Do not include extra commentary.

Itinerary:
{itinerary}

Cheapest flights (JSON):
{json.dumps(cheapest_flights)}

Context:
{json.dumps(context)}
"""
    try:
        out = call_gemini_text(prompt, max_output_tokens=800)
        if not out:
            return None
        # Try robust JSON extraction (strip code fences/markdown)
        cleaned = out.strip()
        # Remove triple backticks and language markers
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
        # If text contains surrounding non-json, try to find the first '[' ... ']' block
        if not (cleaned.startswith("[") and cleaned.endswith("]")):
            m = re.search(r"(\[.*\])", cleaned, flags=re.DOTALL)
            if m:
                cleaned = m.group(1)
        packages = json.loads(cleaned)
        # Ensure totals are numbers
        for p in packages:
            if "total_cost" not in p:
                total = sum(int(p.get(k, 0) or 0) for k in ["estimated_hotel_cost","estimated_food_cost","estimated_transport_cost","activity_cost","flight_cost"])
                p["total_cost"] = total
        return packages
    except Exception as e:
        print("⚠️ Gemini packages parse error:", e)
        return None

# -------------------------
# Routes
# -------------------------
@app.get("/")
def root():
    return {"message": "🌍 Welcome to RoamGenie API — check /api/health"}

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.post("/api/contact")
def api_contact(payload: ContactPayload):
    try:
        record = {
            "name": f"{payload.first_name.strip()} {payload.last_name.strip()}".strip(),
            "email": payload.email,
            "phone": payload.phone,
            "message": payload.message,
            "source": payload.source,
            "timestamp": datetime.utcnow().isoformat()
        }
        if supabase:
            res = supabase.table("contacts").insert(record).execute()
            return {"success": True, "data": getattr(res, "data", None)}
        else:
            return {"success": True, "data": record}
    except Exception as e:
        print("❌ contact error:", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/log_search")
def api_log_search(payload: SearchPayload):
    record = {
        "user_id": payload.user_id,
        "departure_city": payload.departure_city,
        "destination": payload.destination,
        "days": payload.days,
        "theme": payload.theme,
        "activities": payload.activities,
        "budget_preference": payload.budget_preference,
        "travel_class": payload.travel_class,
        "departure_date": payload.departure_date,
        "return_date": payload.return_date,
        "timestamp": datetime.utcnow().isoformat(),
    }
    try:
        if supabase:
            res = supabase.table("search_logs").insert(record).execute()
            return {"success": True, "data": getattr(res, "data", None)}
        else:
            return {"success": True, "data": record}
    except Exception as e:
        print("❌ DB insert error:", e)
        raise HTTPException(status_code=500, detail=str(e))

# Passport OCR + optional upload to Supabase storage (if supabase available)
@app.post("/api/scan_passport")
async def scan_passport(file: UploadFile = File(...)):
    if not PIL_AVAILABLE:
        raise HTTPException(status_code=500, detail="OCR libraries missing")
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        # Basic OCR
        text = pytesseract.image_to_string(img)
        passport_no = None
        nationality = None
        for line in text.split("\n"):
            if "passport" in line.lower():
                passport_no = line.split()[-1]
            if "nationality" in line.lower():
                nationality = line.split()[-1]
        # Optional: upload file to supabase storage if configured
        public_url = None
        if supabase:
            safe_filename = re.sub(r"[^a-zA-Z0-9_.-]", "_", file.filename)
            file_path = f"passport_uploads/{safe_filename}"
            try:
                bucket = supabase.storage.from_("passport-images")
                bucket.upload(path=file_path, file=contents, file_options={"upsert": "true"})
                public_url = bucket.get_public_url(file_path)
            except Exception as e:
                print("⚠️ Supabase upload failed:", e)
        return {
            "success": True,
            "passport_number": passport_no,
            "nationality": nationality,
            "raw_text": text,
            "public_url": public_url
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Simple visa-free CSV-based lookup (public dataset)
@app.get("/api/visa_free/{country}")
def get_visa_free_for_country(country: str):
    try:
        country = country.strip()
        url = "https://raw.githubusercontent.com/ilyankou/passport-index-dataset/master/passport-index-tidy.csv"
        df = requests.get(url, timeout=15).content.decode("utf-8")
        import io as _io, csv as _csv
        rdr = _csv.DictReader(_io.StringIO(df))
        matches = []
        for row in rdr:
            if row.get("Passport", "").strip().lower() == country.lower():
                req = row.get("Requirement", "")
                if "visa free" in req.lower() or "visa on arrival" in req.lower():
                    matches.append(row.get("Destination"))
        destinations = sorted(list(set([d for d in matches if d])))
        return {"passport_country": country, "visa_free_countries": destinations, "count": len(destinations)}
    except Exception as e:
        print("❌ Error fetching visa-free list:", e)
        raise HTTPException(status_code=500, detail=str(e))

# Generate itinerary (single unified implementation)
@app.post("/api/generate_itinerary")
def api_generate_itinerary(payload: GeneratePayload):
    try:
        serp_json = {}
        cheapest_flights = []
        if SERPAPI_KEY and payload.departure_city:
            serp_json = fetch_flights_serpapi(payload.departure_city, payload.destination, payload.departure_date or "", payload.return_date or "")
            try:
                cheapest_flights = normalize_flights_from_serp(serp_json) if serp_json else []
            except Exception:
                cheapest_flights = []

        cheapest_trains = []
        if RAPIDAPI_KEY and payload.departure_date:
            try:
                dep_date_obj = datetime.strptime(payload.departure_date, "%Y-%m-%d")
                trains_json = fetch_trains_rapidapi(payload.departure_city or "", payload.destination or "", payload.departure_date)
                cheapest_trains = normalize_trains_from_rapid(trains_json, dep_date_obj.date()) if trains_json else []
            except Exception:
                cheapest_trains = []

        hotels_text = ""
        if GEMINI_ENABLED:
            try:
                hotels_prompt = f"Provide 3-6 bullet points of recommended hotels and restaurants in {payload.destination} suitable for mid-range travellers."
                hotels_text = call_gemini_text(hotels_prompt, max_output_tokens=250) or ""
            except Exception as e:
                print("⚠️ hotels gemini failed:", e)
                hotels_text = f"Sample hotel & restaurant suggestions for {payload.destination}. (Gemini failed; see logs.)"
        else:
            hotels_text = f"Sample hotel & restaurant suggestions for {payload.destination}. (Enable Gemini for richer results.)"

        itinerary_text = None
        if GEMINI_ENABLED:
            try:
                prompt = f"Create a structured {payload.days}-day Markdown itinerary for {payload.destination}. Theme: {payload.theme}. Activities: {payload.activities}. Use the cheapest transport options: {json.dumps(cheapest_flights + cheapest_trains)}. Hotel info: {hotels_text}\nReturn only Markdown."
                itinerary_text = call_gemini_text(prompt, max_output_tokens=1200)
            except Exception as e:
                print("⚠️ Gemini itinerary generation failed:", e)
                itinerary_text = None

        if not itinerary_text:
            itinerary_text = fallback_create_itinerary(payload, cheapest_flights, cheapest_trains, hotels_text)

        return {
            "success": True,
            "itinerary": itinerary_text,
            "cheapest_flights": cheapest_flights,
            "cheapest_trains": cheapest_trains,
            "hotel_restaurant_content": hotels_text,
        }
    except Exception as e:
        print("❌ generate_itinerary error:", e)
        raise HTTPException(status_code=500, detail=str(e))

# Modify itinerary
@app.post("/api/modify_itinerary")
def api_modify_itinerary(payload: ModifyPayload):
    try:
        updated_text = None
        if GEMINI_ENABLED:
            try:
                prompt = f"""
You are an assistant that edits travel itineraries. Here is the current itinerary:

{payload.current_itinerary}

User requested modification:
{payload.modification_prompt}

Please return a revised itinerary in Markdown, reflecting the user's modification. Preserve original structure and clearly indicate the changes you made.
"""
                updated_text = call_gemini_text(prompt, max_output_tokens=800)
            except Exception as e:
                print("⚠️ Gemini modify failed:", e)
                updated_text = None

        if not updated_text:
            updated_text = payload.current_itinerary + "\n\n---\n### Modification Requested\n" + payload.modification_prompt + "\n\n✔️ Updated above."

        return {"success": True, "updated_itinerary": updated_text}
    except Exception as e:
        print("❌ modify_itinerary error:", e)
        raise HTTPException(status_code=500, detail=str(e))

# Finalize packages (uses Gemini when available, else fallback)
@app.post("/api/finalize_packages")
def api_finalize_packages(payload: PackagePayload):
    try:
        packages = None
        cheapest_flights = payload.context.get("cheapest_flights", []) if isinstance(payload.context, dict) else []
        if GEMINI_ENABLED:
            try:
                packages = call_gemini_packages(payload.itinerary, payload.context or {}, cheapest_flights)
            except Exception as e:
                print("⚠️ Gemini packages failed:", e)
                packages = None

        if not packages:
            packages = fallback_generate_packages(payload.itinerary, payload.context or {}, cheapest_flights)

        return {"success": True, "packages": packages}
    except Exception as e:
        print("❌ finalize_packages error:", e)
        raise HTTPException(status_code=500, detail=str(e))



if st.session_state.current_page == "Passport":
    # ... (all your existing passport page code) ...
    st.markdown("""
        <div class="passport-scan">
            <h2 style="color: white; text-align: center;">Passport Scanner</h2>
            <p style="color: white; text-align: center;">Upload your passport photo to discover visa-free travel destinations!</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Upload Passport Photo")
        uploaded_file = st.file_uploader(
            "Choose passport image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear photo of your passport's main page"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Passport", use_container_width=True)

            if st.button("Scan Passport"):
                with st.spinner("Uploading and scanning passport..."):
                    uploaded_file.seek(0)
                    response = requests.post(
                        "http://127.0.0.1:8000/api/scan_passport",
                        files={"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    )
                    if response.status_code == 200:
                        result = response.json()
                        if result.get("country_detected"):
                            country = result["country_detected"]
                            confidence = result.get("confidence", 0)
                            st.session_state.passport_country = country
                            st.success(f"Detected passport country: {country} (Confidence: {confidence:.1%})")

                            with st.spinner("Finding visa-free destinations..."):
                                visa_free = passport_scanner.get_visa_free_countries(country)
                                st.session_state.visa_free_countries = visa_free
                        else:
                            st.error("Could not detect country from passport image.")
                    else:
                        st.error(f"API error: {response.text}")


    with col2:
        st.subheader("Manual Country Selection")
        st.write("Or select your passport country manually:")

        available_countries = []
        if passport_scanner.visa_data is not None:
            available_countries = sorted(passport_scanner.visa_data['Passport'].unique().tolist())
        else:
            available_countries = [
                'India', 'United States', 'United Kingdom', 'Germany', 'France',
                'Singapore', 'Japan', 'Australia', 'Canada', 'Netherlands',
                'Switzerland', 'Sweden', 'Norway', 'Denmark', 'Finland'
            ]

        selected_country = st.selectbox("Select your passport country:", [''] + available_countries)

        if selected_country and st.button("Find Visa-Free Countries"):
            st.session_state.passport_country = selected_country
            with st.spinner("Finding visa-free destinations..."):
                visa_free = passport_scanner.get_visa_free_countries(selected_country)
                st.session_state.visa_free_countries = visa_free

    if st.session_state.passport_country and st.session_state.visa_free_countries:
        st.markdown("---")
        st.subheader(f"Visa-Free Destinations for {st.session_state.passport_country} Passport Holders")
        st.info(f"Great news! You can travel to {len(st.session_state.visa_free_countries)} countries visa-free!")
        search_country = st.text_input("Search countries:", placeholder="Type to filter countries...")

        filtered_countries = st.session_state.visa_free_countries
        if search_country:
            filtered_countries = [country for country in st.session_state.visa_free_countries
                                    if search_country.lower() in country.lower()]
            st.write(f"Found {len(filtered_countries)} countries matching '{search_country}'")

        if filtered_countries:
            cols = st.columns(4)
            for idx, country in enumerate(filtered_countries):
                col_idx = idx % 4
                with cols[col_idx]:
                    flag = passport_scanner.country_flags.get(country, '')
                    st.markdown(f"""
                        <div class="visa-free-card">
                            <div class="country-name">{flag} {country}</div>
                            <div class="visa-status">Visa-Free Entry</div>
                        </div>
                    """, unsafe_allow_html=True)
        else:
            st.write("No countries found matching your search.")

        st.markdown("### Regional Breakdown")

        # ... (Regional breakdown lists remain unchanged) ...
        asia_countries = ['Thailand', 'Singapore', 'Malaysia', 'Indonesia', 'Philippines',]
        europe_countries = ['Germany', 'France', 'Italy', 'Spain', 'United Kingdom', 'Ireland',]
        middle_east_countries = ['UAE', 'Qatar', 'Oman', 'Kuwait', 'Bahrain', 'Saudi Arabia',]
        africa_countries = ['Mauritius', 'Seychelles', 'Madagascar', 'Comoros', 'Cape Verde',]
        americas_countries = ['United States', 'Canada', 'Mexico', 'Brazil', 'Argentina',]
        oceania_countries = ['Australia', 'New Zealand', 'Fiji', 'Vanuatu', 'Samoa', 'Tonga',]


        asia_count = sum(1 for country in st.session_state.visa_free_countries if country in asia_countries)
        europe_count = sum(1 for country in st.session_state.visa_free_countries if country in europe_countries)
        middle_east_count = sum(1 for country in st.session_state.visa_free_countries if country in middle_east_countries)
        africa_count = sum(1 for country in st.session_state.visa_free_countries if country in africa_countries)
        americas_count = sum(1 for country in st.session_state.visa_free_countries if country in americas_countries)
        oceania_count = sum(1 for country in st.session_state.visa_free_countries if country in oceania_countries)

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.metric("Asia", asia_count)
        with col2:
            st.metric("Europe", europe_count)
        with col3:
            st.metric("Middle East", middle_east_count)
        with col4:
            st.metric("Africa", africa_count)
        with col5:
            st.metric("Americas", americas_count)
        with col6:
            st.metric("Oceania", oceania_count)


# ✅ FINAL FIXED CONTACT US SECTION
if st.session_state.current_page == "Contact Us":
    st.header("📞 Contact Us")
    st.write("We’d love to hear from you! Please fill out the form below and we’ll get back to you soon.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Your Information")
        first_name = st.text_input("First Name")
        last_name = st.text_input("Last Name")
        email = st.text_input("Email")
        phone = st.text_input("Phone Number")
        message = st.text_area("Message", "I'm interested in travel planning services...")

        if st.button("📩 Submit", use_container_width=True):
            if not first_name or not email or not phone:
                st.warning("⚠️ Please fill out all required fields (First Name, Email, Phone).")
            else:
                try:
                    # ✅ Save contact to Supabase
                    log_contact(
                        name=f"{first_name} {last_name}".strip(),
                        email=email,
                        message=message,
                        phone=phone,
                        source="contact_us_form"
                    )

                    st.success("✅ Your message has been submitted successfully!")

                except Exception as e:
                    st.error(f"❌ Failed to save contact: {e}")

    with col2:
        st.subheader("🎯 Quick Actions")
        if st.button("🔄 Reset Session"):
            for key in list(st.session_state.keys()):
                if key not in ['session_id']:
                    del st.session_state[key]
            st.success("Session reset successfully!")
            st.rerun()
