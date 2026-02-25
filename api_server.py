# =========================================================
#  RoamGenie Backend — FIXED TRAIN API VERSION
# =========================================================

import os
import json
import re
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
    allow_origins=[
        "https://roam-genie-drab.vercel.app",
        "https://roam-genie-frontend.vercel.app"
    ],
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

def format_date_range(departure_date: str, return_date: str, days: int) -> str:
    """Format the date range nicely for display"""
    try:
        dep = datetime.strptime(departure_date, "%Y-%m-%d")
        ret = datetime.strptime(return_date, "%Y-%m-%d")
        return f"{dep.strftime('%B %d, %Y')} – {ret.strftime('%B %d, %Y')} ({days} Days)"
    except:
        return f"{departure_date} to {return_date} ({days} days)"

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

    for f in best_flights[:3]:
        legs = f.get("flights", [])
        if not legs:
            continue

        first = legs[0]
        last = legs[-1]

        airline = first.get("airline") or "Unknown Airline"
        logo = first.get("airline_logo") or ""

        dep = first.get("departure_airport", {})
        arr = last.get("arrival_airport", {})

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
# Trains - FIXED WITH FREE API
# =========================================================

def fetch_live_trains_free(from_code: str, to_code: str, date_str: str = None):
    """
    Uses IRCTC1 API v3 endpoint - trainBetweenStations
    This is the CORRECT endpoint that works with free tier subscription
    """
    if not RAPIDAPI_KEY:
        print("⚠️ No RAPIDAPI_KEY found")
        return {"quota_exceeded": True}
    
    # Use current date if not provided
    if not date_str:
        from datetime import datetime
        date_str = datetime.now().strftime("%Y-%m-%d")
    
    try:
        # CORRECT WORKING ENDPOINT for free tier
        url = "https://irctc1.p.rapidapi.com/api/v3/trainBetweenStations"
        
        querystring = {
            "fromStationCode": from_code,
            "toStationCode": to_code,
            "dateOfJourney": date_str
        }
        
        headers = {
            "X-RapidAPI-Key": RAPIDAPI_KEY,
            "X-RapidAPI-Host": "irctc1.p.rapidapi.com"
        }
        
        print(f"🚂 Calling IRCTC1 API v3: {from_code} → {to_code} on {date_str}")
        
        response = requests.get(url, headers=headers, params=querystring, timeout=15)
        
        print(f"📡 API Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Train API Success! Data: {json.dumps(data)[:300]}...")
            return data
        elif response.status_code == 403:
            print(f"⚠️ 403 Error - Check if you subscribed to the IRCTC1 API on RapidAPI")
            print(f"   Go to: https://rapidapi.com/IRCTCAPI/api/irctc1")
            print(f"   Click 'Subscribe to Test' and select FREE plan")
            return {"quota_exceeded": True}
        elif response.status_code == 429:
            print(f"⚠️ Rate limit exceeded (free tier: 20 calls/month)")
            return {"quota_exceeded": True}
        else:
            print(f"⚠️ API Error: Status {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error details: {error_data}")
            except:
                print(f"   Response: {response.text[:200]}")
            return {"quota_exceeded": True}
            
    except Exception as e:
        print(f"❌ Exception calling Train API: {e}")
        import traceback
        traceback.print_exc()
        return {"quota_exceeded": True}

def get_fallback_trains(from_code: str, to_code: str):
    """
    Comprehensive fallback train data for major Indian routes
    """
    print(f"🚂 Using fallback trains for {from_code} → {to_code}")
    
    # Major routes database
    routes = {
        # Mumbai - Delhi
        ("BOM", "DEL"): [
            {
                "type": "Train",
                "provider": "Rajdhani Express (12951)",
                "provider_logo": "🚄",
                "price": "₹2,500 - ₹4,000",
                "departure_time": "16:55",
                "arrival_time": "08:35",
                "duration_minutes": "15h 40m",
                "booking_link": "https://www.irctc.co.in/nget/train-search"
            },
            {
                "type": "Train",
                "provider": "August Kranti Rajdhani (12953)",
                "provider_logo": "🚄",
                "price": "₹2,400 - ₹3,800",
                "departure_time": "17:20",
                "arrival_time": "09:10",
                "duration_minutes": "15h 50m",
                "booking_link": "https://www.irctc.co.in/nget/train-search"
            },
            {
                "type": "Train",
                "provider": "Paschim Express (12925)",
                "provider_logo": "🚂",
                "price": "₹1,200 - ₹2,500",
                "departure_time": "22:45",
                "arrival_time": "17:00",
                "duration_minutes": "18h 15m",
                "booking_link": "https://www.irctc.co.in/nget/train-search"
            }
        ],
        
        # Delhi - Mumbai
        ("DEL", "BOM"): [
            {
                "type": "Train",
                "provider": "Mumbai Rajdhani (12952)",
                "provider_logo": "🚄",
                "price": "₹2,500 - ₹4,000",
                "departure_time": "16:00",
                "arrival_time": "08:35",
                "duration_minutes": "16h 35m",
                "booking_link": "https://www.irctc.co.in/nget/train-search"
            },
            {
                "type": "Train",
                "provider": "August Kranti Rajdhani (12954)",
                "provider_logo": "🚄",
                "price": "₹2,400 - ₹3,800",
                "departure_time": "16:55",
                "arrival_time": "10:05",
                "duration_minutes": "17h 10m",
                "booking_link": "https://www.irctc.co.in/nget/train-search"
            }
        ],
        
        # Mumbai - Bangalore
        ("BOM", "BLR"): [
            {
                "type": "Train",
                "provider": "Udyan Express (11301)",
                "provider_logo": "🚂",
                "price": "₹1,500 - ₹3,000",
                "departure_time": "08:05",
                "arrival_time": "07:50",
                "duration_minutes": "23h 45m",
                "booking_link": "https://www.irctc.co.in/nget/train-search"
            }
        ],
        
        # Delhi - Bangalore
        ("DEL", "BLR"): [
            {
                "type": "Train",
                "provider": "Karnataka Express (12628)",
                "provider_logo": "🚂",
                "price": "₹1,800 - ₹3,500",
                "departure_time": "19:15",
                "arrival_time": "05:45",
                "duration_minutes": "34h 30m",
                "booking_link": "https://www.irctc.co.in/nget/train-search"
            }
        ],
        
        # Delhi - Kolkata
        ("DEL", "KOL"): [
            {
                "type": "Train",
                "provider": "Rajdhani Express (12301)",
                "provider_logo": "🚄",
                "price": "₹2,200 - ₹3,800",
                "departure_time": "16:55",
                "arrival_time": "10:05",
                "duration_minutes": "17h 10m",
                "booking_link": "https://www.irctc.co.in/nget/train-search"
            }
        ],
        
        # Mumbai - Kolkata
        ("BOM", "KOL"): [
            {
                "type": "Train",
                "provider": "Gitanjali Express (12859)",
                "provider_logo": "🚂",
                "price": "₹1,600 - ₹3,200",
                "departure_time": "06:10",
                "arrival_time": "13:50",
                "duration_minutes": "31h 40m",
                "booking_link": "https://www.irctc.co.in/nget/train-search"
            }
        ]
    }
    
    # Return route-specific data or generic fallback
    specific_route = routes.get((from_code, to_code))
    if specific_route:
        return specific_route
    
    # Generic fallback for any route
    return [
        {
            "type": "Train",
            "provider": f"Express Train {from_code}-{to_code}",
            "provider_logo": "🚂",
            "price": "₹1,000 - ₹2,500",
            "departure_time": "08:00",
            "arrival_time": "18:00",
            "duration_minutes": "10h 00m",
            "booking_link": "https://www.irctc.co.in/nget/train-search"
        },
        {
            "type": "Train",
            "provider": f"Superfast Train {from_code}-{to_code}",
            "provider_logo": "🚄",
            "price": "₹1,500 - ₹3,000",
            "departure_time": "14:30",
            "arrival_time": "06:30",
            "duration_minutes": "16h 00m",
            "booking_link": "https://www.irctc.co.in/nget/train-search"
        },
        {
            "type": "Train",
            "provider": f"Mail Express {from_code}-{to_code}",
            "provider_logo": "🚂",
            "price": "₹800 - ₹2,000",
            "departure_time": "22:00",
            "arrival_time": "12:00",
            "duration_minutes": "14h 00m",
            "booking_link": "https://www.irctc.co.in/nget/train-search"
        }
    ]

def normalize_trains(response: dict, from_code: str = "", to_code: str = ""):
    """
    Normalizes train results from IRCTC1 API v3 (trainBetweenStations endpoint)
    Expected response format:
    {
        "status": true,
        "message": "Success",
        "timestamp": 1234567890,
        "data": [
            {
                "train_number": "12951",
                "train_name": "MUMBAI RAJDHANI",
                "train_src": "NDLS",
                "train_dstn": "BCT",
                "from_sta": "NDLS",
                "from_std": "16:55",
                "to_sta": "08:35",
                "train_date": "2025-11-28",
                "duration": "15:40",
                "run_days": ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"],
                ...
            }
        ]
    }
    """
    # Check if API failed or quota exceeded
    if response.get("quota_exceeded"):
        print("⚠️ API quota exceeded, using fallback trains")
        return get_fallback_trains(from_code, to_code)
    
    # Check for unsuccessful status
    if not response.get("status"):
        msg = response.get("message", "Unknown error")
        print(f"⚠️ API returned unsuccessful status: {msg}")
        return get_fallback_trains(from_code, to_code)
    
    # Check if we have train data
    trains = response.get("data", [])
    
    if not trains or len(trains) == 0:
        print("⚠️ No trains in API response, using fallback")
        return get_fallback_trains(from_code, to_code)

    # Normalize API response
    results = []
    print(f"📋 Processing {len(trains)} trains from API...")
    
    for t in trains[:3]:  # Top 3 trains
        train_name = t.get("train_name", "Unknown Train")
        train_number = t.get("train_number", "")
        
        # Duration format
        duration = t.get("duration", "N/A")
        
        # Get departure and arrival times
        dept_time = t.get("from_std", "N/A")
        arr_time = t.get("to_sta", "N/A")
        
        # Price estimation based on train type
        price = "₹800 - ₹2,000"  # Default
        train_upper = train_name.upper()
        
        if "RAJDHANI" in train_upper or "SHATABDI" in train_upper:
            price = "₹2,000 - ₹4,000"
        elif "DURONTO" in train_upper or "GARIB RATH" in train_upper:
            price = "₹1,500 - ₹3,000"
        elif "VANDE BHARAT" in train_upper:
            price = "₹1,800 - ₹3,500"
        elif "EXPRESS" in train_upper or "MAIL" in train_upper:
            price = "₹800 - ₹2,000"
        elif "SUPERFAST" in train_upper:
            price = "₹1,000 - ₹2,500"
        
        results.append({
            "type": "Train",
            "provider": f"{train_name} ({train_number})",
            "provider_logo": "🚂",
            "price": price,
            "departure_time": dept_time,
            "arrival_time": arr_time,
            "duration_minutes": duration,
            "booking_link": "https://www.irctc.co.in/nget/train-search"
        })
        
        print(f"   ✓ {train_name} ({train_number}): {dept_time} → {arr_time} | {duration}")

    if results:
        print(f"✅ Successfully normalized {len(results)} trains from LIVE API!")
        return results
    
    # Final fallback
    print("⚠️ No valid trains after normalization, using fallback")
    return get_fallback_trains(from_code, to_code)

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
            cleaned = cleaned.strip("`").replace("json", "").strip()
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
    """Create a basic HTML itinerary when Gemini is unavailable"""
    date_range = format_date_range(
        payload.departure_date or "2025-01-01",
        payload.return_date or "2025-01-03",
        payload.days or 2
    )
    
    return f"""
<div style="font-family: system-ui, -apple-system, sans-serif; color: #1e293b; line-height: 1.8;">
  <h1 style="font-size: 2.25rem; font-weight: 700; margin-bottom: 1rem; color: #ef4444;">
    {payload.theme or 'Trip'} in {payload.destination}
  </h1>
  
  <p style="font-size: 1.125rem; line-height: 1.75; margin-bottom: 2rem; color: #64748b;">
    A wonderful {payload.days}-day adventure awaits you in {payload.destination}!
  </p>
  
  <div style="background: #fef2f2; padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 2rem; border-left: 4px solid #ef4444;">
    <h2 style="font-size: 1.5rem; font-weight: 700; margin-bottom: 1rem; color: #1e293b;">Travel Summary</h2>
    <ul style="list-style: none; padding: 0; line-height: 2;">
      <li><strong style="color: #1e293b;">Destination:</strong> {payload.destination}</li>
      <li><strong style="color: #1e293b;">Duration:</strong> {date_range}</li>
      <li><strong style="color: #1e293b;">Theme:</strong> {payload.theme or 'General'}</li>
      <li><strong style="color: #1e293b;">Departure City:</strong> {payload.departure_city or 'N/A'}</li>
    </ul>
  </div>
  
  <h2 style="font-size: 1.75rem; font-weight: 700; margin: 2rem 0 1rem; color: #ef4444;">Day-by-Day Itinerary</h2>
  
  <div style="margin-bottom: 2rem;">
    <h3 style="font-size: 1.25rem; font-weight: 700; color: #ef4444; margin-bottom: 0.75rem;">Day 1: Arrival & Exploration</h3>
    <ul style="line-height: 1.8; color: #475569;">
      <li><strong style="color: #1e293b;">Morning:</strong> Arrive in {payload.destination} and check into your hotel</li>
      <li><strong style="color: #1e293b;">Afternoon:</strong> Explore the local area and get oriented</li>
      <li><strong style="color: #1e293b;">Evening:</strong> Enjoy dinner at a local restaurant</li>
    </ul>
  </div>
  
  <div style="background: #fef3c7; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #f59e0b; margin-top: 2rem;">
    <h3 style="font-size: 1.25rem; font-weight: 700; margin-bottom: 0.5rem; color: #92400e;">💡 Travel Tips</h3>
    <ul style="line-height: 1.8;">
      <li>Book accommodations in advance for better rates</li>
      <li>Keep digital copies of important documents</li>
      <li>Try local cuisine for an authentic experience</li>
    </ul>
  </div>
</div>
"""

# =========================================================
# API Endpoints
# =========================================================
@app.get("/api/campaign/{theme}")
def api_get_campaign(theme: str):
    try:
        # Fetch active campaign matching theme
        data = supabase.table("campaigns").select("*") \
            .eq("theme", theme.lower()) \
            .eq("active", True) \
            .maybe_single()

        if data and data.data:
            return {"success": True, "campaign": data.data}

        return {"success": True, "campaign": None}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate_itinerary")
def api_generate_itinerary(payload: GeneratePayload):
    try:
        # Fetch flights
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

        # Fetch trains with fallback
        cheapest_trains = []
        if payload.departure_city and payload.destination:
            try:
                print(f"🚂 Fetching trains: {payload.departure_city} → {payload.destination}")
                trains_json = fetch_live_trains_free(
                    payload.departure_city,
                    payload.destination,
                    payload.departure_date or datetime.now().strftime("%Y-%m-%d")
                )
                cheapest_trains = normalize_trains(
                    trains_json, 
                    payload.departure_city, 
                    payload.destination
                )
                print(f"✅ Got {len(cheapest_trains)} trains")
            except Exception as e:
                print(f"⚠️ Train fetch error: {e}, using fallback")
                cheapest_trains = get_fallback_trains(payload.departure_city, payload.destination)

        # Fetch hotels
        hotels_text = ""
        if GEMINI_ENABLED:
            h_prompt = f"""Give 3–6 recommended hotels and restaurants in {payload.destination}.
            
IMPORTANT: Do NOT use any asterisk symbols. Use plain text only. Do not use markdown formatting."""
            hotels_text = call_gemini_text(h_prompt, 200) or ""
            
            if hotels_text:
                hotels_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', hotels_text)
                hotels_text = hotels_text.replace('**', '').replace('*', '')
        else:
            hotels_text = f"Hotel suggestions for {payload.destination}."
        

        # -----------------------------------------------------------
        # Fetch Campaign (if theme matches)
        # -----------------------------------------------------------
        campaign = None
        try:
            theme = (payload.theme or "").lower()
            camp = (
                supabase.table("campaigns")
                .select("*")
                .eq("theme", theme)
                .eq("active", True)
                .maybe_single()
                .execute()
            )

            if camp.data:
                campaign = camp.data[0] if isinstance(camp.data, list) else camp.data
                print("🎉 Campaign found for itinerary:", campaign)

        except Exception as e:
            print("⚠️ Campaign lookup failed:", e)

        # Generate HTML itinerary
        itinerary_html = None
        if GEMINI_ENABLED:
            date_range = format_date_range(
                payload.departure_date or "2025-01-01",
                payload.return_date or "2025-01-03",
                payload.days or 2
            )
            
            prompt = f"""
You are RoamGenie, an expert travel planner.

Generate a beautifully formatted HTML itinerary with proper styling.

CRITICAL RULES:
- Return ONLY HTML content, NO markdown, NO code fences, NO backticks
- DO NOT use any asterisk symbols: NO **, NO *, NO markdown
- Use ONLY proper HTML tags: <h1>, <h2>, <h3>, <p>, <ul>, <li>, <div>, <strong>, <span>
- Use inline CSS styles for all formatting
- For bold text, use <strong style="color: #1e293b;">text</strong>

COLOR PALETTE (RED THEME):
- Primary: #ef4444 (red for headings)
- Secondary: #dc2626 (dark red)
- Text: #1e293b (dark slate)
- Muted: #64748b
- Background: #fef2f2 (light red tint)

TRIP DATA:
Destination: {payload.destination}
Days: {payload.days}
Theme: {payload.theme}
Activities: {payload.activities or 'General sightseeing'}
Departure City: {payload.departure_city}
Date Range: {date_range}

Hotels/Restaurants:
{hotels_text}

Generate complete HTML itinerary now. Remember: PURE HTML only, no markdown, NO asterisk symbols!
"""
            itinerary_html = call_gemini_text(prompt, 1500)

            if itinerary_html:
                itinerary_html = itinerary_html.replace("```html", "").replace("```", "").strip()
                itinerary_html = re.sub(r'\*\*([^*]+)\*\*', r'<strong style="color: #1e293b;">\1</strong>', itinerary_html)
                itinerary_html = itinerary_html.replace('**', '').replace('*', '')

        if not itinerary_html:
            itinerary_html = fallback_create_itinerary(payload, cheapest_flights, cheapest_trains, hotels_text)

        return {
            "success": True,
            "itinerary": itinerary_html,
            "cheapest_flights": cheapest_flights,
            "cheapest_trains": cheapest_trains,
            "hotel_restaurant_content": hotels_text,
            "campaign": campaign,   # ⭐ ADD THIS
        }


    except Exception as e:
        print("❌ generate_itinerary error:", e)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/modify_itinerary")
def api_modify_itinerary(payload: ModifyPayload):
    try:
        modified = None
        if GEMINI_ENABLED:
            prompt = f"""
Edit the following HTML itinerary based on the user's request.
Maintain the same HTML structure and styling.
Return ONLY the updated HTML, no markdown, no code fences.

Current Itinerary:
{payload.current_itinerary}

User's Request:
{payload.modification_prompt}

Return the complete updated HTML itinerary.
"""
            modified = call_gemini_text(prompt, 1200)
            
            if modified:
                modified = modified.replace("```html", "").replace("```", "").strip()

        if not modified:
            modified = (
                payload.current_itinerary
                + "<div style='background: #fef3c7; padding: 1rem; border-radius: 0.5rem; margin-top: 2rem;'>"
                + "<h3>Modification Request:</h3>"
                + f"<p>{payload.modification_prompt}</p>"
                + "</div>"
            )

        return {"success": True, "updated_itinerary": modified}

    except Exception as e:
        print("❌ modify_itinerary error:", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/finalize_packages")
def api_finalize_packages(payload: PackagePayload):
    print(">>> Finalizing packages <<<")

    try:
        print("📦 RAW CONTEXT:", payload.context)

        # Normalize keys
        if isinstance(payload.context, dict):
            payload.context = {k.lower(): v for k, v in payload.context.items()}

        print("📦 CLEAN CONTEXT:", payload.context)
        print("🔑 Available Keys:", list(payload.context.keys()))

        # Check nested trip data
        trip_ctx = payload.context.get("trip") or payload.context.get("route") or {}
        print("📍 TRIP CTX:", trip_ctx)

        # Extract cities (flat or nested)
        from_city = (
            payload.context.get("departure_city")
            or payload.context.get("from_city")
            or trip_ctx.get("from_city")
            or trip_ctx.get("origin")
            or ""
        ).strip().upper()

        to_city = (
            payload.context.get("destination")
            or payload.context.get("to_city")
            or trip_ctx.get("to_city")
            or trip_ctx.get("destination")
            or ""
        ).strip().upper()

        print(f"🧪 Extracted route: {from_city} → {to_city}")

        # Query Supabase
        fetched = (
            supabase.table("packages")
            .select("*")
            .ilike("from_city", f"%{from_city}%")
            .ilike("to_city", f"%{to_city}%")
            .eq("active", True)
            .execute()
        )

        packages = fetched.data or []
        print("📦 Found packages:", len(packages))

        if len(packages) == 0:
            return {
                "success": True,
                "packages": [],
                "call_us": True,
                "message": "No packages available for this route."
            }

        # Apply discount (unchanged)
        def to_int(val):
            try:
                return int(str(val).replace("₹", "").replace(",", ""))
            except:
                return 0

        theme = payload.context.get("theme", "").lower()

        campaign_data = (
            supabase.table("campaigns")
            .select("*")
            .eq("theme", theme)
            .eq("active", True)
            .maybe_single()
            .execute()
        )

        if campaign_data.data:
            campaign = campaign_data.data
            discount_percent = campaign["discount_percent"]

            for p in packages:
                original = to_int(p["price"])
                discount_amount = int(original * discount_percent / 100)
                p["discount_percent"] = discount_percent
                p["discount_amount"] = discount_amount
                p["discounted_total"] = original - discount_amount

        return {"success": True, "packages": packages}

    except Exception as e:
        print("❌ finalize_packages error:", e)
        raise HTTPException(status_code=500, detail=str(e))





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


@app.get("/")
def read_root():
    return {"message": "RoamGenie API is running!", "status": "ok"}

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

        # Normalize and return country_detected
        country_detected = None
        confidence = 0.80

        # Simple mapping example
        COUNTRY_KEYWORDS = {
            "INDIA": "India",
            "USA": "United States",
            "UNITED STATES": "United States",
            "UK": "United Kingdom",
            "UNITED KINGDOM": "United Kingdom",
            "JAPAN": "Japan",
            "CANADA": "Canada",
            "AUSTRALIA": "Australia",
            "SINGAPORE": "Singapore",
            "MALAYSIA": "Malaysia",
            "THAILAND": "Thailand",
            "UAE": "United Arab Emirates",
            "GERMANY": "Germany",
            "FRANCE": "France",
            "ITALY": "Italy",
            "SPAIN": "Spain",
            "BRAZIL": "Brazil",
            "MEXICO": "Mexico",
            "CHINA": "China",
            "SOUTH KOREA": "South Korea",
            "INDONESIA": "Indonesia"
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
            "country_detected": country_detected,
            "confidence": confidence,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================================================
# ADDITIONAL TRAIN ENDPOINT FOR TESTING
# =========================================================

@app.get("/api/trains/test")
def test_train_endpoint(from_code: str = "NDLS", to_code: str = "BCT", date: str = None):
    """
    Test endpoint to verify train functionality
    Usage: GET /api/trains/test?from_code=NDLS&to_code=BCT&date=2025-12-01
    """
    try:
        if not date:
            from datetime import datetime
            date = datetime.now().strftime("%Y-%m-%d")
        
        print(f"🧪 Testing train fetch: {from_code} → {to_code} on {date}")
        
        # Try API first
        trains_json = fetch_live_trains_free(from_code, to_code, date)
        
        # Normalize with fallback
        trains = normalize_trains(trains_json, from_code, to_code)
        
        return {
            "success": True,
            "from": from_code,
            "to": to_code,
            "date": date,
            "trains": trains,
            "api_response": trains_json,
            "fallback_used": trains_json.get("quota_exceeded", False),
            "note": "If you see 403 error, subscribe to free plan at: https://rapidapi.com/IRCTCAPI/api/irctc1"
        }
    except Exception as e:
        print(f"❌ Test endpoint error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "fallback_trains": get_fallback_trains(from_code, to_code)
        }

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
