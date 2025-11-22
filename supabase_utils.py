from supabase import create_client
import os
import streamlit as st

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# 🧭 Save trip search log
def log_search(destination, budget_preference, travel_class, travel_days, user_id="guest"):
    try:
        supabase.table("search_logs").insert({
            "user_id": user_id,
            "destination": destination,
            "budget_preference": budget_preference,
            "travel_class": travel_class,
            "travel_days": travel_days
        }).execute()
    except Exception as e:
        st.error(f"❌ Failed to log search: {e}")

# 💬 Save contact form
def log_contact(name, email, message, phone=None, source="web_form"):
    try:
        supabase.table("contacts").insert({
            "name": name,
            "email": email,
            "message": message,
            "phone": phone,
            "source": source
        }).execute()
    except Exception as e:
        st.error(f"❌ Failed to save contact: {e}")

# 📊 Fetch analytics summary
def get_analytics():
    try:
        total_searches = supabase.table("search_logs").select("id", count="exact").execute().count
        total_contacts = supabase.table("contacts").select("id", count="exact").execute().count
        
        budget_data = supabase.rpc("get_popular_budget").execute()
        popular_budget = budget_data.data[0]["budget_preference"] if budget_data.data else "N/A"

        class_data = supabase.rpc("get_popular_class").execute()
        popular_class = class_data.data[0]["travel_class"] if class_data.data else "N/A"

        return {
            "total_searches": total_searches,
            "total_contacts": total_contacts,
            "popular_budget": popular_budget,
            "popular_class": popular_class
        }
    except Exception as e:
        st.error(f"❌ Failed to fetch analytics: {e}")
        return {}

