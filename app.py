import os
import json
from typing import Any

import streamlit as st
from google import genai
from google.genai import types

MODEL_NAME = "gemini-3.6-flash"

st.set_page_config(page_title="Mobile Phone Information & Price Finder", page_icon="📱", layout="wide")


def get_api_key() -> str:
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
        if key:
            return str(key).strip()
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY", "").strip()


def get_client() -> genai.Client:
    key = get_api_key()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it to Streamlit Secrets.")
    return genai.Client(api_key=key)


PHONE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "found": {"type": "BOOLEAN"},
        "model": {"type": "STRING"},
        "brand": {"type": "STRING"},
        "release_date": {"type": "STRING"},
        "network": {"type": "STRING"},
        "sim": {"type": "STRING"},
        "display": {
            "type": "OBJECT",
            "properties": {
                "type": {"type": "STRING"}, "size": {"type": "STRING"},
                "resolution": {"type": "STRING"}, "refresh_rate": {"type": "STRING"},
                "protection": {"type": "STRING"}
            },
            "required": ["type", "size", "resolution", "refresh_rate", "protection"]
        },
        "performance": {
            "type": "OBJECT",
            "properties": {
                "chipset": {"type": "STRING"}, "cpu": {"type": "STRING"},
                "gpu": {"type": "STRING"}, "ram": {"type": "STRING"},
                "storage": {"type": "STRING"}, "os": {"type": "STRING"}
            },
            "required": ["chipset", "cpu", "gpu", "ram", "storage", "os"]
        },
        "camera": {
            "type": "OBJECT",
            "properties": {
                "rear": {"type": "STRING"}, "front": {"type": "STRING"},
                "video": {"type": "STRING"}, "features": {"type": "STRING"}
            },
            "required": ["rear", "front", "video", "features"]
        },
        "battery": {
            "type": "OBJECT",
            "properties": {
                "capacity": {"type": "STRING"}, "charging": {"type": "STRING"},
                "wireless_charging": {"type": "STRING"}
            },
            "required": ["capacity", "charging", "wireless_charging"]
        },
        "connectivity": {
            "type": "OBJECT",
            "properties": {
                "wifi": {"type": "STRING"}, "bluetooth": {"type": "STRING"},
                "nfc": {"type": "STRING"}, "usb": {"type": "STRING"}, "gps": {"type": "STRING"}
            },
            "required": ["wifi", "bluetooth", "nfc", "usb", "gps"]
        },
        "other_features": {"type": "STRING"},
        "pakistan_price": {
            "type": "OBJECT",
            "properties": {
                "official_or_expected": {"type": "STRING"},
                "pta_approved": {"type": "STRING"},
                "non_pta": {"type": "STRING"},
                "currency": {"type": "STRING"},
                "price_note": {"type": "STRING"}
            },
            "required": ["official_or_expected", "pta_approved", "non_pta", "currency", "price_note"]
        },
        "lahore_availability": {
            "type": "OBJECT",
            "properties": {
                "availability": {"type": "STRING"}, "typical_stores": {"type": "STRING"}, "note": {"type": "STRING"}
            },
            "required": ["availability", "typical_stores", "note"]
        },
        "karachi_availability": {
            "type": "OBJECT",
            "properties": {
                "availability": {"type": "STRING"}, "typical_stores": {"type": "STRING"}, "note": {"type": "STRING"}
            },
            "required": ["availability", "typical_stores", "note"]
        },
        "notes": {"type": "STRING"}
    },
    "required": [
        "found", "model", "brand", "release_date", "network", "sim", "display",
        "performance", "camera", "battery", "connectivity", "other_features",
        "pakistan_price", "lahore_availability", "karachi_availability", "notes"
    ]
}


PROMPT = """
You are a mobile-phone information assistant for a user in Pakistan.

IMPORTANT:
- Use ONLY your built-in/model knowledge.
- DO NOT use Google Search, web browsing, grounding, URL context, or any external tool.
- This application intentionally makes ONE Gemini API call per search.
- Never claim you checked a live website, shop inventory, or today's price.
- For Pakistan prices, PTA status, and Lahore/Karachi availability, give your best
  knowledge-based estimate and clearly mention uncertainty or possible outdated data.
- Never invent exact current store stock.
- If you do not know something reliably, write "Not available".
- Return only the requested structured JSON.
"""


def research_mobile(model_name: str) -> dict[str, Any]:
    client = get_client()
    user_prompt = f"""{PROMPT}

Requested phone model: {model_name}

Provide:
1. Basic model information
2. Display
3. Performance
4. Cameras
5. Battery and charging
6. Connectivity
7. Other important features
8. Pakistan price, including PTA-approved and non-PTA where known
9. General availability guidance for Lahore and Karachi
10. Important notes and uncertainty
"""

    # EXACTLY ONE Gemini API call. No Google Search tool is configured.
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=PHONE_SCHEMA,
            temperature=0.2,
        ),
    )

    if not response.text:
        raise ValueError("Gemini returned an empty response.")
    if response.parsed:
        return response.parsed
    return json.loads(response.text)


def safe_value(value: Any) -> str:
    if value is None or value == "":
        return "Not available"
    if isinstance(value, list):
        return ", ".join(str(x) for x in value) if value else "Not available"
    return str(value)


def show_field(label: str, value: Any) -> None:
    st.markdown(f"**{label}:** {safe_value(value)}")


st.title("📱 Mobile Phone Information & Price Finder")
st.caption("Gemini-only mode — one Gemini API call per search. No Google Search is used.")
st.info("Because this version does not access the web, prices, PTA status, and store availability are knowledge-based and may not be current.")

with st.form("mobile_search"):
    model_name = st.text_input("Enter mobile model", placeholder="e.g. Samsung Galaxy S25 Ultra")
    submitted = st.form_submit_button("🔎 Search Mobile", use_container_width=True)

if submitted:
    if not model_name.strip():
        st.warning("Please enter a mobile model.")
        st.stop()
    if not get_api_key():
        st.error("Gemini API key is missing. Add GEMINI_API_KEY to Streamlit Secrets.")
        st.stop()

    with st.spinner("Gemini is preparing the phone information..."):
        try:
            data = research_mobile(model_name.strip())
        except Exception as exc:
            msg = str(exc)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                st.error("Gemini API rate limit/quota was reached. Wait a little and try again, or check your Gemini API quota.")
            elif "401" in msg or "403" in msg:
                st.error("The Gemini API key is invalid or does not have access to this project/model.")
            elif "404" in msg or "NOT_FOUND" in msg:
                st.error(f"The selected model ({MODEL_NAME}) was not found for this API key/project.")
            else:
                st.error(f"Could not retrieve the information: {msg}")
            st.stop()

    if not data.get("found", True):
        st.warning("Gemini could not confidently identify this phone model. Please check the model name.")
        st.stop()

    st.success(f"Information for {safe_value(data.get('model'))}")

    st.subheader("📌 Overview")
    c1, c2, c3 = st.columns(3)
    with c1:
        show_field("Brand", data.get("brand")); show_field("Model", data.get("model"))
    with c2:
        show_field("Release", data.get("release_date")); show_field("Network", data.get("network"))
    with c3:
        show_field("SIM", data.get("sim")); show_field("Other Features", data.get("other_features"))

    st.subheader("🖥️ Display")
    d = data.get("display", {})
    c1, c2 = st.columns(2)
    with c1:
        show_field("Type", d.get("type")); show_field("Size", d.get("size")); show_field("Resolution", d.get("resolution"))
    with c2:
        show_field("Refresh Rate", d.get("refresh_rate")); show_field("Protection", d.get("protection"))

    st.subheader("⚙️ Performance")
    p = data.get("performance", {})
    c1, c2 = st.columns(2)
    with c1:
        show_field("Chipset", p.get("chipset")); show_field("CPU", p.get("cpu")); show_field("GPU", p.get("gpu"))
    with c2:
        show_field("RAM", p.get("ram")); show_field("Storage", p.get("storage")); show_field("Operating System", p.get("os"))

    st.subheader("📷 Camera")
    cam = data.get("camera", {})
    c1, c2 = st.columns(2)
    with c1:
        show_field("Rear Camera", cam.get("rear")); show_field("Front Camera", cam.get("front"))
    with c2:
        show_field("Video", cam.get("video")); show_field("Features", cam.get("features"))

    st.subheader("🔋 Battery")
    b = data.get("battery", {})
    c1, c2, c3 = st.columns(3)
    with c1: show_field("Capacity", b.get("capacity"))
    with c2: show_field("Charging", b.get("charging"))
    with c3: show_field("Wireless Charging", b.get("wireless_charging"))

    st.subheader("📡 Connectivity")
    n = data.get("connectivity", {})
    c1, c2, c3 = st.columns(3)
    with c1: show_field("Wi-Fi", n.get("wifi"))
    with c2: show_field("Bluetooth", n.get("bluetooth"))
    with c3: show_field("NFC", n.get("nfc"))
    c1, c2 = st.columns(2)
    with c1: show_field("USB", n.get("usb"))
    with c2: show_field("GPS", n.get("gps"))

    st.subheader("🇵🇰 Pakistan Price")
    price = data.get("pakistan_price", {})
    c1, c2 = st.columns(2)
    with c1:
        show_field("Official / Expected", price.get("official_or_expected"))
        show_field("PTA Approved", price.get("pta_approved"))
        show_field("Non-PTA", price.get("non_pta"))
    with c2:
        show_field("Currency", price.get("currency")); show_field("Price Note", price.get("price_note"))

    st.subheader("📍 Lahore Availability")
    lahore = data.get("lahore_availability", {})
    show_field("Availability", lahore.get("availability")); show_field("Typical Stores / Areas", lahore.get("typical_stores")); show_field("Note", lahore.get("note"))

    st.subheader("📍 Karachi Availability")
    karachi = data.get("karachi_availability", {})
    show_field("Availability", karachi.get("availability")); show_field("Typical Stores / Areas", karachi.get("typical_stores")); show_field("Note", karachi.get("note"))

    st.subheader("📝 Important Notes")
    st.write(safe_value(data.get("notes")))

    st.caption("Gemini generated this result without Google Search or live web access. Verify prices, PTA status, and stock before purchase.")
