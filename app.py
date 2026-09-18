import os
import json
import time
from typing import Any

import streamlit as st
from google import genai
from google.genai import types

MODEL_NAME = "gemini-3.6-flash"

st.set_page_config(
    page_title="Mobile Phone Information & Price Finder",
    page_icon="📱",
    layout="wide",
)

def get_api_key() -> str:
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
        if key:
            return str(key).strip()
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY", "").strip()

def get_client() -> genai.Client:
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it to Streamlit Secrets."
        )
    return genai.Client(api_key=api_key)

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
                "type": {"type": "STRING"},
                "size": {"type": "STRING"},
                "resolution": {"type": "STRING"},
                "refresh_rate": {"type": "STRING"},
                "protection": {"type": "STRING"},
            },
            "required": ["type", "size", "resolution", "refresh_rate", "protection"],
        },
        "performance": {
            "type": "OBJECT",
            "properties": {
                "chipset": {"type": "STRING"},
                "cpu": {"type": "STRING"},
                "gpu": {"type": "STRING"},
                "ram": {"type": "STRING"},
                "storage": {"type": "STRING"},
                "os": {"type": "STRING"},
            },
            "required": ["chipset", "cpu", "gpu", "ram", "storage", "os"],
        },
        "camera": {
            "type": "OBJECT",
            "properties": {
                "rear": {"type": "STRING"},
                "front": {"type": "STRING"},
                "video": {"type": "STRING"},
                "features": {"type": "STRING"},
            },
            "required": ["rear", "front", "video", "features"],
        },
        "battery": {
            "type": "OBJECT",
            "properties": {
                "capacity": {"type": "STRING"},
                "charging": {"type": "STRING"},
                "wireless_charging": {"type": "STRING"},
            },
            "required": ["capacity", "charging", "wireless_charging"],
        },
        "connectivity": {
            "type": "OBJECT",
            "properties": {
                "wifi": {"type": "STRING"},
                "bluetooth": {"type": "STRING"},
                "nfc": {"type": "STRING"},
                "usb": {"type": "STRING"},
                "gps": {"type": "STRING"},
            },
            "required": ["wifi", "bluetooth", "nfc", "usb", "gps"],
        },
        "other_features": {"type": "STRING"},
        "pakistan_price": {
            "type": "OBJECT",
            "properties": {
                "official_or_expected": {"type": "STRING"},
                "pta_approved": {"type": "STRING"},
                "non_pta": {"type": "STRING"},
                "currency": {"type": "STRING"},
                "price_note": {"type": "STRING"},
            },
            "required": [
                "official_or_expected",
                "pta_approved",
                "non_pta",
                "currency",
                "price_note",
            ],
        },
        "lahore_availability": {
            "type": "OBJECT",
            "properties": {
                "availability": {"type": "STRING"},
                "typical_stores": {"type": "STRING"},
                "note": {"type": "STRING"},
            },
            "required": ["availability", "typical_stores", "note"],
        },
        "karachi_availability": {
            "type": "OBJECT",
            "properties": {
                "availability": {"type": "STRING"},
                "typical_stores": {"type": "STRING"},
                "note": {"type": "STRING"},
            },
            "required": ["availability", "typical_stores", "note"],
        },
        "notes": {"type": "STRING"},
    },
    "required": [
        "found", "model", "brand", "release_date", "network", "sim",
        "display", "performance", "camera", "battery", "connectivity",
        "other_features", "pakistan_price", "lahore_availability",
        "karachi_availability", "notes",
    ],
}

SYSTEM_PROMPT = """
You are a mobile-phone information assistant for users in Pakistan.

Use ONLY your built-in/model knowledge.
DO NOT use Google Search, web browsing, grounding, or any external tool.
This application is intentionally designed around one Gemini API call per search.

Never claim that you checked a live website, shop inventory, or today's price.
For Pakistan prices, PTA status, and store availability, provide your best
knowledge-based information and clearly state that it may be outdated.
Do not invent exact current shop stock.
If something is not reliably known, write "Not available".
Consider differences between RAM/storage variants, PTA/non-PTA status,
official/imported units, and seller pricing.

Return only structured JSON matching the supplied schema.
"""

def research_mobile(model_name: str) -> dict[str, Any]:
    """One normal Gemini call. Retries only if the service temporarily returns 503/429."""
    client = get_client()

    prompt = f"""
{SYSTEM_PROMPT}

Requested mobile model:
{model_name}

Create a Pakistan-focused report with:
- basic model information
- display
- performance
- camera
- battery and charging
- connectivity
- other features
- Pakistan price information
- PTA-approved and non-PTA information
- general Lahore availability
- general Karachi availability
- important notes and uncertainty

Do not present live prices or live inventory as verified facts.
"""

    last_error = None

    # Normal case: exactly one API call.
    # The extra attempts happen only after temporary 429/503 service errors.
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
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

        except Exception as exc:
            last_error = exc
            message = str(exc)

            if ("503" in message or "UNAVAILABLE" in message or
                    "429" in message or "RESOURCE_EXHAUSTED" in message):
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                    continue
            raise

    raise last_error


def safe_value(value: Any) -> str:
    if value is None or value == "":
        return "Not available"
    if isinstance(value, list):
        return ", ".join(str(x) for x in value) if value else "Not available"
    return str(value)


def field(label: str, value: Any) -> None:
    st.markdown(f"**{label}:** {safe_value(value)}")


def two_col_fields(left_items, right_items):
    left, right = st.columns(2)
    with left:
        for label, value in left_items:
            field(label, value)
    with right:
        for label, value in right_items:
            field(label, value)


# ---------------- UI ----------------

st.title("📱 Mobile Phone Information & Price Finder")
st.caption(
    "Gemini-only mode • One Gemini API call per search • No Google Search"
)

st.info(
    "⚠️ This version does not access the web. Prices, PTA information, and "
    "store availability are based on Gemini's knowledge and may not be current."
)

with st.form("mobile_search"):
    model_name = st.text_input(
        "Enter mobile model",
        placeholder="e.g. Samsung Galaxy S25 Ultra",
    )
    submitted = st.form_submit_button(
        "🔎 Search Mobile",
        use_container_width=True,
    )

if submitted:
    if not model_name.strip():
        st.warning("Please enter a mobile model.")
        st.stop()

    if not get_api_key():
        st.error(
            "GEMINI_API_KEY is missing. Add it under Streamlit → Settings → Secrets."
        )
        st.stop()

    with st.spinner("Gemini is preparing the phone information..."):
        try:
            data = research_mobile(model_name.strip())
        except Exception as exc:
            message = str(exc)

            if "503" in message or "UNAVAILABLE" in message:
                st.error(
                    "Gemini is temporarily experiencing high demand (503). "
                    "Please wait a little and search again."
                )
            elif "429" in message or "RESOURCE_EXHAUSTED" in message:
                st.error(
                    "Gemini API rate/quota limit was reached (429). "
                    "Please wait and try again."
                )
            elif "401" in message or "403" in message:
                st.error(
                    "The Gemini API key is invalid or does not have access "
                    "to this model/project."
                )
            elif "404" in message or "NOT_FOUND" in message:
                st.error(
                    f"The selected model ({MODEL_NAME}) was not found for this project."
                )
            else:
                st.error(f"Could not retrieve the information: {message}")
            st.stop()

    if not data.get("found", True):
        st.warning(
            "Gemini could not confidently identify this phone model. "
            "Please check the model name."
        )
        st.stop()

    st.success(f"Information for {safe_value(data.get('model'))}")

    # Every section is a separate dropdown on the same page.
    with st.expander("📌 1. Overview", expanded=True):
        two_col_fields(
            [
                ("Brand", data.get("brand")),
                ("Model", data.get("model")),
                ("Release Date", data.get("release_date")),
            ],
            [
                ("Network", data.get("network")),
                ("SIM", data.get("sim")),
                ("Other Features", data.get("other_features")),
            ],
        )

    with st.expander("🖥️ 2. Display"):
        display = data.get("display", {})
        two_col_fields(
            [
                ("Display Type", display.get("type")),
                ("Screen Size", display.get("size")),
                ("Resolution", display.get("resolution")),
            ],
            [
                ("Refresh Rate", display.get("refresh_rate")),
                ("Protection", display.get("protection")),
            ],
        )

    with st.expander("⚙️ 3. Performance"):
        performance = data.get("performance", {})
        two_col_fields(
            [
                ("Chipset", performance.get("chipset")),
                ("CPU", performance.get("cpu")),
                ("GPU", performance.get("gpu")),
            ],
            [
                ("RAM", performance.get("ram")),
                ("Storage", performance.get("storage")),
                ("Operating System", performance.get("os")),
            ],
        )

    with st.expander("📷 4. Camera"):
        camera = data.get("camera", {})
        two_col_fields(
            [
                ("Rear Camera", camera.get("rear")),
                ("Front Camera", camera.get("front")),
            ],
            [
                ("Video", camera.get("video")),
                ("Camera Features", camera.get("features")),
            ],
        )

    with st.expander("🔋 5. Battery & Charging"):
        battery = data.get("battery", {})
        left, right, extra = st.columns(3)
        with left:
            field("Battery Capacity", battery.get("capacity"))
        with right:
            field("Charging", battery.get("charging"))
        with extra:
            field("Wireless Charging", battery.get("wireless_charging"))

    with st.expander("📡 6. Connectivity"):
        connectivity = data.get("connectivity", {})
        left, middle, right = st.columns(3)
        with left:
            field("Wi-Fi", connectivity.get("wifi"))
            field("USB", connectivity.get("usb"))
        with middle:
            field("Bluetooth", connectivity.get("bluetooth"))
            field("GPS", connectivity.get("gps"))
        with right:
            field("NFC", connectivity.get("nfc"))

    with st.expander("🇵🇰 7. Pakistan Price & PTA"):
        price = data.get("pakistan_price", {})
        two_col_fields(
            [
                ("Official / Expected Price", price.get("official_or_expected")),
                ("PTA Approved", price.get("pta_approved")),
                ("Non-PTA", price.get("non_pta")),
            ],
            [
                ("Currency", price.get("currency")),
                ("Price Note", price.get("price_note")),
            ],
        )

    with st.expander("📍 8. Lahore Availability"):
        lahore = data.get("lahore_availability", {})
        field("General Availability", lahore.get("availability"))
        field("Typical Stores / Areas", lahore.get("typical_stores"))
        field("Note", lahore.get("note"))

    with st.expander("📍 9. Karachi Availability"):
        karachi = data.get("karachi_availability", {})
        field("General Availability", karachi.get("availability"))
        field("Typical Stores / Areas", karachi.get("typical_stores"))
        field("Note", karachi.get("note"))

    with st.expander("📝 10. Important Notes"):
        st.write(safe_value(data.get("notes")))

    st.warning(
        "Before purchasing, verify the current price, PTA status, warranty, "
        "RAM/storage variant, and actual stock with the seller. This app "
        "does not perform live web searches."
    )
