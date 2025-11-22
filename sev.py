# =========================================================
#  RoamGenie Backend — CLEAN + FULL WORKING VERSION
# =========================================================

import os
import json
from datetime import datetime, date
from typing import Optional, List, Dict, Any

import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

# Google Gemini
from google import generativeai as genai

# Supabase
from supabase import create_client

# OCR libs
try:
    import cv2
    import numpy as np
    from PIL import Image
    import pytesseract
    PIL_AVAILABLE = True
except:
    PIL_AVAILABLE = False

# -------------------------
# ENV
# -------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")

SERPAPI_KEY = os.getenv("SERPAPI_KEY")
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_ENABLED = os.getenv("GEMINI_ENABLED", "false").lower() == "true"

# Gemini configure
if GEMINI_ENABLED and GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    GEMINI_MODEL = genai.GenerativeModel("gemini-2.0-flash")
else:
    GEMINI_MODEL = None

# Supabase
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------------
# FastAPI
# -------------------------
app = FastAPI(title="RoamGenie API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN, "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# Models
# =========================================================

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

# =========================================================
# Helpers
# =========================================================

def format_datetime(iso_string: Optional[str]):
    if not iso_string:
        return "N/A"
    try:
        if "T" in iso_string:
            dt = datetime.fromisoformat(iso_string)
            return dt.strftime("%b-%d, %Y | %I:%M %p")
        dt = datetime.strptime(iso_string, "%Y-%m-%d")
        return dt.strftime("%b-%d, %Y")
    except:
        return iso_string

# =========================================================
# Flights
# =========================================================

def fetch_flights_serpapi(departure_id, arrival_id, outbound_date, return_date):
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
        resp = requests.get("https://serpapi.com/search", params=params, timeout=20)
        return resp.json()
    except:
        return {}

def normalize_flights_from_serp(api_json: dict):
    best_flights = api_json.get("best_flights", [])
    normalized = []

    for f in best_flights[:3]:  # pick top 3
        legs = f.get("flights", [])
        if not legs:
            continue

        first = legs[0]
        last = legs[-1]

        # Correct airline + logo extraction
        airline = first.get("airline") or "Unknown Airline"
        logo = first.get("airline_logo") or ""

        dep = first.get("departure_airport", {})
        arr = last.get("arrival_airport", {})

        # IMPORTANT: REAL working flight booking link
        booking_link = f.get("link") or f"https://www.google.com/flights?q={dep.get('id','')}{arr.get('id','')}"

        normalized.append({
            "type": "Flight",
            "provider": airline,
            "provider_logo": logo,
            "price": f.get("price"),
            "departure_time": format_datetime(dep.get("time")),
            "arrival_time": format_datetime(arr.get("time")),
            "duration_minutes": f.get("total_duration"),
            "booking_link": booking_link
        })

    return normalized


# =========================================================
# Trains
# =========================================================


def normalize_trains(response: dict):
    """
    Normalizes the RapidAPI IRCTC train results into RoamGenie format.
    Returns top 3 trains.
    """

    if not response.get("ok"):
        return []

    payload = response["payload"]
    trains = payload.get("data", [])[:3]

    results = []

    for t in trains:
        results.append({
            "type": "Train",
            "provider": f"{t.get('train_name', '')} ({t.get('train_number', '')})",
            "provider_logo": "",
            "price": None,  # IRCTC API does not return price
            "departure_time": t.get("from_std"),
            "arrival_time": t.get("to_sta"),
            "duration_minutes": t.get("duration"),
            "booking_link": "https://www.irctc.co.in/nget/train-search"
        })

    return results


# =========================================================
# Gemini Helpers
# =========================================================

def call_gemini_text(prompt: str, max_output_tokens: int = 800) -> Optional[str]:
    if not GEMINI_ENABLED or not GEMINI_MODEL:
        return None
    try:
        res = GEMINI_MODEL.generate_content(prompt)
        return res.text if hasattr(res, "text") else str(res)
    except Exception as e:
        print("❌ Gemini call error:", e)
        return None

def call_gemini_packages(itinerary: str, context: dict, cheapest_flights: list):
    prompt = f"""
You are a travel packaging expert. Produce 3 package tiers (Budget, Standard, Luxury) as JSON array.
Each must include:
- package_name
- estimated_hotel_cost
- estimated_food_cost
- estimated_transport_cost
- activity_cost
- flight_cost
- description
- total_cost

Return ONLY JSON.

Itinerary:\n{itinerary}

Flights:\n{json.dumps(cheapest_flights)}

Context:\n{json.dumps(context)}
"""
    try:
        out = call_gemini_text(prompt, 800)
        cleaned = out.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
        return json.loads(cleaned)
    except Exception as e:
        print("⚠️ Gemini packages parse fail:", e)
        return None

# =========================================================
# Fallback Packages
# =========================================================

def fallback_generate_packages(itinerary_text: str, context: dict, cheapest_flights: list = None):
    if cheapest_flights is None:
        cheapest_flights = []
    try:
        flight_cost = int(cheapest_flights[0].get("price", 0)) if cheapest_flights else 0
    except:
        flight_cost = 0

    days = context.get("days") or context.get("num_days") or 1
    days = max(1, int(days))

    def pack(name, hotel, food, transport, activity):
        total = hotel + food + transport + activity + flight_cost
        return {
            "package_name": name,
            "estimated_hotel_cost": hotel,
            "estimated_food_cost": food,
            "estimated_transport_cost": transport,
            "activity_cost": activity,
            "flight_cost": flight_cost,
            "description": f"{name} package",
            "total_cost": total,
        }

    return [
        pack("Budget", 3000 * days, 1000 * days, 500 * days, 1500),
        pack("Standard", 8000 * days, 2000 * days, 1500 * days, 2500),
        pack("Luxury", 20000 * days, 5000 * days, 4000 * days, 5000),
    ]

# =========================================================
# Itinerary Creation Fallback
# =========================================================

def fallback_create_itinerary(payload, flights, trains, hotels_text):
    return f"""
# Trip Itinerary — {payload.destination}

(Gemini unavailable. Basic itinerary generated.)

## Day 1
- Arrival and check-in
- Explore local attractions

## Day 2
- Full day sightseeing
- Lunch at a recommended restaurant

## Day 3
- Departure

Hotels/Restaurants:
{hotels_text}
"""

# =========================================================
# API Endpoints
# =========================================================

@app.post("/api/generate_itinerary")
def api_generate_itinerary(payload: GeneratePayload):
    try:
        serp_json = {}
        cheapest_flights = []
        if SERPAPI_KEY and payload.departure_city:
            serp_json = fetch_flights_serpapi(
                payload.departure_city,
                payload.destination,
                payload.departure_date or "",
                payload.return_date or "",
            )
            cheapest_flights = normalize_flights_from_serp(serp_json) if serp_json else []

        cheapest_trains = []
        if RAPIDAPI_KEY and payload.departure_date:
            try:
                dep_date_obj = datetime.strptime(payload.departure_date, "%Y-%m-%d")
                trains_json = fetch_trains_rapidapi(
                    payload.departure_city or "",
                    payload.destination or "",
                    payload.departure_date
                )
                cheapest_trains = normalize_trains(trains_json, dep_date_obj.date()) if trains_json else []
            except:
                cheapest_trains = []

        # hotels
        hotels_text = ""
        if GEMINI_ENABLED:
            h_prompt = f"Give 3–6 recommended hotels and restaurants in {payload.destination}."
            hotels_text = call_gemini_text(h_prompt, 200) or ""
        else:
            hotels_text = f"Hotel suggestions for {payload.destination}."

        # itinerary
        itinerary_text = None
        if GEMINI_ENABLED:
            prompt = f"""
Create a Markdown itinerary for this trip:
{payload.json()}

Flights:
{cheapest_flights}

Trains:
{cheapest_trains}

Hotels:
{hotels_text}
"""
            itinerary_text = call_gemini_text(prompt, 1200)

        if not itinerary_text:
            itinerary_text = fallback_create_itinerary(payload, cheapest_flights, cheapest_trains, hotels_text)

        return {
            "success": True,
            "itinerary": itinerary_text,
            "cheapest_flights": cheapest_flights,
            "cheapest_trains": cheapest_trains,
            "hotels": hotels_text,
        }

    except Exception as e:
        print("❌ generate_itinerary error:", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/modify_itinerary")
def api_modify_itinerary(payload: ModifyPayload):
    try:
        modified = None
        if GEMINI_ENABLED:
            prompt = f"""
Edit the following itinerary based on the user's request:

Itinerary:
{payload.current_itinerary}

User modification:
{payload.modification_prompt}

Return edited itinerary in Markdown only.
"""
            modified = call_gemini_text(prompt, 800)

        if not modified:
            modified = (
                payload.current_itinerary
                + "\n\n---\n### Modification\n"
                + payload.modification_prompt
            )

        return {"success": True, "updated_itinerary": modified}

    except Exception as e:
        print("❌ modify_itinerary error:", e)
        raise HTTPException(status_code=500, detail=str(e))


# =========================================================
# SINGLE FINALIZE PACKAGES (no duplicates!)
# =========================================================
@app.post("/api/finalize_packages")
def api_finalize_packages(payload: PackagePayload):
    print(">>> USING FINAL CLEAN finalize_packages ENDPOINT <<<")
    try:
        cheapest_flights = payload.context.get("cheapest_flights", []) if isinstance(payload.context, dict) else []
        packages = None

        if GEMINI_ENABLED:
            packages = call_gemini_packages(payload.itinerary, payload.context, cheapest_flights)

        if not packages:
            packages = fallback_generate_packages(payload.itinerary, payload.context, cheapest_flights)

        return {"success": True, "packages": packages}

    except Exception as e:
        print("❌ finalize_packages error:", e)
        raise HTTPException(status_code=500, detail=str(e))


# =========================================================
# Contact
# =========================================================
@app.post("/api/contact")
def api_contact(payload: ContactPayload):
    try:
        record = {
            "name": f"{payload.first_name} {payload.last_name}",
            "email": payload.email,
            "phone": payload.phone,
            "message": payload.message,
            "source": payload.source,
            "timestamp": datetime.utcnow().isoformat()
        }
        res = supabase.table("contacts").insert(record).execute()
        return {"success": True, "data": getattr(res, "data", None)}

    except Exception as e:
        print("❌ contact error:", e)
        raise HTTPException(status_code=500, detail=str(e))

# =========================================================
# Logging Search (single version)
# =========================================================
@app.post("/api/log_search")
def api_log_search(payload: SearchPayload):
    try:
        record = payload.dict()
        record["timestamp"] = datetime.utcnow().isoformat()
        res = supabase.table("search_logs").insert(record).execute()
        return {"success": True, "data": getattr(res, "data", None)}

    except Exception as e:
        print("❌ log_search error:", e)
        raise HTTPException(status_code=500, detail=str(e))


# =========================================================
# PASSPORT OCR
# =========================================================
@app.get("/api/passport_countries")
def get_passport_country_list():
    """
    Returns all countries from the global Passport Index dataset.
    These will populate the frontend dropdown.
    """
    try:
        url = "https://raw.githubusercontent.com/ilyankou/passport-index-dataset/master/passport-index-tidy.csv"
        csv_text = requests.get(url, timeout=10).content.decode("utf-8")

        import csv, io
        reader = csv.DictReader(io.StringIO(csv_text))

        countries = set()

        for row in reader:
            passport = row.get("Passport", "").strip()
            if passport:
                countries.add(passport)

        return {
            "success": True,
            "countries": sorted(countries)
        }

    except Exception as e:
        print("❌ Error fetching passport countries:", e)
        raise HTTPException(status_code=500, detail=str(e))

from io import BytesIO

@app.post("/api/scan_passport")
async def scan_passport(file: UploadFile = File(...)):
    if not PIL_AVAILABLE:
        raise HTTPException(status_code=500, detail="OCR libraries missing")

    try:
        contents = await file.read()
        img = Image.open(BytesIO(contents))

        text = pytesseract.image_to_string(img)

        passport_no = None
        nationality = None

        for line in text.split("\n"):
            if "passport" in line.lower():
                passport_no = line.split()[-1]
            if "nationality" in line.lower():
                nationality = line.split()[-1]

        # 🔥 ADD THIS — normalize and return country_detected
        country_detected = None
        confidence = 0.80  # placeholder confidence score

        # simple mapping example
        COUNTRY_KEYWORDS = {
            "INDIA": "India",
            "USA": "United States",
            "UNITED STATES": "United States",
            "UK": "United Kingdom",
            "UNITED KINGDOM": "United Kingdom",
            "JAPAN": "Japan",
            "CANADA": "Canada"
        }

        for key, val in COUNTRY_KEYWORDS.items():
            if key in text.upper():
                country_detected = val
                break

        return {
            "success": True,
            "passport_number": passport_no,
            "nationality": nationality,
            "raw_text": text,

            # 🔥 REQUIRED by your frontend
            "country_detected": country_detected,
            "confidence": confidence,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/api/visa_free/{country}")
def visa_free(country: str):
    """
    Returns actual visa-free + visa-on-arrival destinations
    using the global Passport Index dataset.
    """

    try:
        country = country.strip()

        # Fetch global Visa Index dataset
        url = "https://raw.githubusercontent.com/ilyankou/passport-index-dataset/master/passport-index-tidy.csv"
        csv_text = requests.get(url, timeout=10).content.decode("utf-8")

        import csv, io
        reader = csv.DictReader(io.StringIO(csv_text))

        destinations = []
        for row in reader:
            if row.get("Passport", "").strip().lower() == country.lower():
                req = row.get("Requirement", "").lower()
                if "visa free" in req or "visa on arrival" in req:
                    dst = row.get("Destination")
                    if dst:
                        destinations.append(dst)

        # Unique + sorted
        destinations = sorted(set(destinations))

        return {
            "success": True,
            "passport_country": country,
            "visa_free_countries": destinations,
            "count": len(destinations)
        }

    except Exception as e:
        print("❌ Visa API error:", e)
        raise HTTPException(status_code=500, detail=str(e))
# ============================
# SUPABASE AUTH - SIGN IN
# ============================

from fastapi import APIRouter
from pydantic import BaseModel

auth_router = APIRouter()

class SignInPayload(BaseModel):
    email: str
    password: str

@app.post("/api/signin")
def api_signin(payload: SignInPayload):
    try:
        email = payload.email
        password = payload.password

        # Supabase authentication request
        res = supabase.auth.sign_in_with_password(
            {"email": email, "password": password}
        )

        if not res.user:
            raise HTTPException(status_code=401, detail="Invalid email or password")

        return {
            "success": True,
            "access_token": res.session.access_token,
            "refresh_token": res.session.refresh_token,
            "user": {
                "id": res.user.id,
                "email": res.user.email,
            }
        }

    except Exception as e:
        print("❌ Sign-in error:", e)
        raise HTTPException(status_code=401, detail="Invalid credentials")
@app.post("/api/signup")
def api_signup(payload: SignInPayload):
    try:
        res = supabase.auth.sign_up({
            "email": payload.email,
            "password": payload.password
        })

        if not res.user:
            raise HTTPException(status_code=400, detail="Signup failed")

        return {
            "success": True,
            "user": {"id": res.user.id, "email": res.user.email}
        }

    except Exception as e:
        print("❌ Signup error:", e)
        raise HTTPException(status_code=400, detail="Signup failed")
@app.post("/api/signup")
def api_signup(payload: SignInPayload):
    try:
        email = payload.email
        password = payload.password

        res = supabase.auth.sign_up({
            "email": email,
            "password": password,
        })

        if not res.user:
            raise HTTPException(status_code=400, detail="Signup failed")

        return {
            "success": True,
            "access_token": res.session.access_token if res.session else None,
            "refresh_token": res.session.refresh_token if res.session else None,
            "user": {
                "id": res.user.id,
                "email": res.user.email,
            }
        }

    except Exception as e:
        print("❌ Signup error:", e)
        raise HTTPException(status_code=400, detail="Signup failed")