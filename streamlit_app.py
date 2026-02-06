import streamlit as st
import requests
import yfinance as yf
from groq import Groq
import os
from dotenv import load_dotenv

# -------------------------------------------------
# LOAD .env
# -------------------------------------------------
load_dotenv()

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(page_title="LLM Agents App", layout="wide")
st.title("🧠 LLM Agents using Groq + Streamlit")

# -------------------------------------------------
# SIDEBAR – API KEYS
# -------------------------------------------------
st.sidebar.header("🔑 API Keys")

groq_key = st.sidebar.text_input(
    "Groq API Key",
    type="password",
    value=os.getenv("GROQ_API_KEY", "")
)

weather_key = st.sidebar.text_input(
    "OpenWeather API Key",
    type="password",
    value=os.getenv("OPENWEATHER_API_KEY", "")
)

exchange_key = st.sidebar.text_input(
    "ExchangeRate API Key",
    type="password",
    value=os.getenv("EXCHANGE_API_KEY", "")
)

# -------------------------------------------------
# LLM
# -------------------------------------------------
def get_llm():
    return Groq(api_key=groq_key)

def llm_response(prompt):
    client = get_llm()
    chat = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
    )
    return chat.choices[0].message.content

# -------------------------------------------------
# WEATHER (WORKING – DO NOT TOUCH)
# -------------------------------------------------
def get_current_weather(city):
    url = (
        "https://api.openweathermap.org/data/2.5/weather"
        f"?q={city}&appid={weather_key}&units=metric"
    )
    return requests.get(url).json()

def get_forecast(city):
    url = (
        "https://api.openweathermap.org/data/2.5/forecast"
        f"?q={city}&appid={weather_key}&units=metric"
    )
    return requests.get(url).json()

# -------------------------------------------------
# EXCHANGE RATE (FIXED AS PER OFFICIAL DOCS)
# -------------------------------------------------
def get_exchange_rates(base_currency):
    url = f"https://v6.exchangerate-api.com/v6/{exchange_key}/latest/{base_currency}"

    try:
        response = requests.get(url, timeout=10)
    except requests.exceptions.RequestException:
        return {"error": "Network error while calling ExchangeRate API"}

    if response.status_code != 200:
        return {"error": f"HTTP error {response.status_code}"}

    if not response.text.strip():
        return {"error": "Empty response from ExchangeRate API"}

    try:
        data = response.json()
    except ValueError:
        return {"error": "Non-JSON response from ExchangeRate API"}

    if data.get("result") != "success":
        return {"error": data.get("error-type", "Unknown API error")}

    return data

# -------------------------------------------------
# UI TABS
# -------------------------------------------------
tab1, tab2 = st.tabs(["🌍 Trip Planner Agent", "💱 Currency & Stock Agent"])

# =================================================
# PART 1 – TRIP PLANNER AGENT
# =================================================
with tab1:
    st.header("✈️ AI Trip Planner")

    city = st.text_input("Destination City")
    days = st.number_input("Trip Duration (days)", 1, 7, 3)
    month = st.selectbox(
        "Month",
        [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
    )

    if st.button("Plan My Trip"):
        if not groq_key or not weather_key:
            st.error("Please provide Groq and OpenWeather API keys.")
        else:
            current = get_current_weather(city)
            forecast_data = get_forecast(city)

            if current.get("cod") != 200:
                st.error("City not found or weather API error.")
            else:
                st.subheader("🌤️ Current Weather")
                st.write(
                    f"{current['weather'][0]['description'].title()}, "
                    f"{current['main']['temp']}°C"
                )

                st.subheader("📅 Weather Forecast")
                shown = 0
                for item in forecast_data.get("list", []):
                    if "12:00:00" in item["dt_txt"] and shown < days:
                        st.write(
                            f"{item['dt_txt']}: "
                            f"{item['main']['temp']}°C, "
                            f"{item['weather'][0]['description']}"
                        )
                        shown += 1

                prompt = f"""
                Plan a {days}-day trip to {city} in {month}.
                Include:
                - Cultural and historical significance
                - Day-wise itinerary
                - Travel tips
                """

                plan = llm_response(prompt)

                st.subheader("📖 Trip Plan")
                st.write(plan)

                st.subheader("🏨 Hotels & Attractions")
                st.markdown(
                    f"[View on Google Maps]"
                    f"(https://www.google.com/maps/search/hotels+in+{city.replace(' ', '+')})"
                )

# =================================================
# PART 2 – CURRENCY & STOCK AGENT
# =================================================
with tab2:
    st.header("💱 Currency & Stock Market Agent")

    country = st.text_input("Enter Country Name")

    if st.button("Get Market Details"):
        if not groq_key or not exchange_key:
            st.error("Please provide Groq and ExchangeRate API keys.")
        else:
            prompt = f"""
            For {country}, provide:
            - Official currency (ISO code only)
            - Major stock exchanges and indices
            - Stock exchange headquarters city
            """

            info = llm_response(prompt)

            st.subheader("📘 Financial Overview")
            st.write(info)

            # Extract currency code safely
            currency = next(
                (w for w in info.split() if len(w) == 3 and w.isupper()),
                None
            )

            if currency:
                rates = get_exchange_rates(currency)

                if "error" in rates:
                    st.error(f"Exchange API Error: {rates['error']}")
                else:
                    st.subheader("💵 Exchange Rates")
                    for cur in ["USD", "INR", "GBP", "EUR"]:
                        st.write(
                            f"1 {currency} → "
                            f"{rates['conversion_rates'].get(cur)} {cur}"
                        )

            index_map = {
                "Japan": "^N225",
                "India": "^BSESN",
                "United States": "^GSPC",
                "US": "^GSPC",
                "UK": "^FTSE",
                "China": "000001.SS",
                "South Korea": "^KS11"
            }

            ticker = index_map.get(country)
            if ticker:
                index = yf.Ticker(ticker)
                st.subheader("📈 Stock Index (Recent)")
                st.dataframe(index.history(period="1d"))

            st.subheader("📍 Stock Exchange HQ")
            st.markdown(
                f"[View on Google Maps]"
                f"(https://www.google.com/maps/search/stock+exchange+{country.replace(' ', '+')})"
            )
