import json
import os
from typing import Any

import streamlit as st
from google import genai
from google.genai import types


st.set_page_config(
    page_title="Mobile Phone Information & Price Finder",
    page_icon="📱",
    layout="wide",
)


def get_api_key() -> str:
    """Read the Gemini key from Streamlit secrets or an environment variable."""
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
        if key:
            return str(key).strip()
    except Exception:
        pass

    return os.getenv("GEMINI_API_KEY", "").strip()


def get_client() -> genai.Client | None:
    key = get_api_key()
    return genai.Client(api_key=key) if key else None


def clean_json(text: str) -> dict[str, Any]:
    """Parse JSON even if Gemini accidentally wraps it in Markdown fences."""
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:] if lines else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("Gemini returned an invalid JSON response.") from exc


def get_grounded_research(model_name: str) -> str:
    """
    Use Gemini + Google Search grounding to collect current web information.

    This is the important part for prices and availability: Gemini is instructed
    to search the live web rather than relying only on its training knowledge.
    """
    client = get_client()
    if client is None:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    prompt = f"""
Research this mobile phone for a Pakistan buyer:

{model_name}

Use Google Search grounding and current web pages.

Your research MUST cover:
1. Official/general specifications.
2. Current or recently published Pakistan prices in PKR.
3. PTA-approved / official-warranty / non-PTA information when available.
4. Retailers or stores serving Lahore.
5. Retailers or stores serving Karachi.
6. Current stock/availability only when the source actually indicates it.
7. Source names and URLs.

Search for Pakistan-specific information. Prefer:
- official manufacturer pages for specifications;
- established Pakistani retailers for prices;
- retailer product pages or clearly dated listings for availability.

Critical rules:
- Do not invent a retailer, price, stock status, location, or URL.
- If live stock cannot be verified, say so.
- Distinguish estimated prices from verified/current listed prices.
- Do not treat an old article as current stock.
- Record the date/context of price information when available.
- PTA status must not be guessed.
- If sources disagree, report the disagreement instead of choosing a number without explanation.

Return a concise research report with source URLs.
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[
                types.Tool(
                    google_search=types.GoogleSearch()
                )
            ]
        ),
    )

    if not response.text:
        raise ValueError("The web research step returned no information.")

    return response.text


def structure_research(model_name: str, research: str) -> dict[str, Any]:
    """Convert the web-grounded research into predictable JSON for the UI."""
    client = get_client()
    if client is None:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    schema = """
{
  "found": true,
  "model": {
    "brand": "",
    "name": "",
    "launch_date": "",
    "operating_system": "",
    "network": "",
    "sim": ""
  },
  "display": {
    "size": "",
    "type": "",
    "resolution": "",
    "refresh_rate": "",
    "protection": ""
  },
  "performance": {
    "processor": "",
    "gpu": "",
    "ram": [],
    "storage": []
  },
  "camera": {
    "rear_main": "",
    "ultrawide": "",
    "telephoto": "",
    "front": "",
    "video": ""
  },
  "battery": {
    "capacity": "",
    "charging": "",
    "wireless_charging": ""
  },
  "features": {
    "5g": "",
    "fingerprint": "",
    "face_unlock": "",
    "water_resistance": "",
    "bluetooth": "",
    "wifi": "",
    "nfc": "",
    "usb": "",
    "weight": "",
    "dimensions": "",
    "colors": []
  },
  "prices_pakistan": [
    {
      "variant": "",
      "pta_status": "",
      "price_pkr": "",
      "price_type": "verified|estimated",
      "source_name": "",
      "source_url": "",
      "date_context": ""
    }
  ],
  "lahore": [
    {
      "store": "",
      "price_pkr": "",
      "availability": "in stock|out of stock|unknown",
      "area": "",
      "source_url": ""
    }
  ],
  "karachi": [
    {
      "store": "",
      "price_pkr": "",
      "availability": "in stock|out of stock|unknown",
      "area": "",
      "source_url": ""
    }
  ],
  "notes": [],
  "sources": [
    {
      "name": "",
      "url": ""
    }
  ]
}
"""

    prompt = f"""
Convert the research below into ONLY valid JSON matching this schema:

{schema}

Mobile model requested:
{model_name}

Research:
--- START RESEARCH ---
{research}
--- END RESEARCH ---

Rules:
- Use only facts supported by the supplied research.
- Never invent missing values.
- Use "Not available" or an empty list when the research does not support a value.
- Do not upgrade an estimated price to verified.
- Do not claim stock is available unless the research supports it.
- Keep source URLs exactly as provided in the research when possible.
- Return JSON only. No Markdown fences.
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        ),
    )

    if not response.text:
        raise ValueError("The formatting step returned no JSON.")

    return clean_json(response.text)


def research_mobile(model_name: str) -> dict[str, Any]:
    research = get_grounded_research(model_name)
    return structure_research(model_name, research)


def safe_value(value: Any) -> str:
    if value is None or value == "":
        return "Not available"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else "Not available"
    return str(value)


def show_field(label: str, value: Any) -> None:
    st.markdown(f"**{label}:** {safe_value(value)}")


def show_store_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        st.info("No reliable store availability could be verified.")
        return

    table = []
    for row in rows:
        table.append(
            {
                "Store": safe_value(row.get("store")),
                "Price (PKR)": safe_value(row.get("price_pkr")),
                "Availability": safe_value(row.get("availability")),
                "Area": safe_value(row.get("area")),
                "Source": safe_value(row.get("source_url")),
            }
        )

    st.dataframe(table, use_container_width=True, hide_index=True)


st.title("📱 Mobile Phone Information & Price Finder")
st.write(
    "Enter a mobile model to view specifications, Pakistan pricing, "
    "and retailer information for Lahore and Karachi."
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
            "Gemini API key is missing. Add GEMINI_API_KEY to Streamlit secrets."
        )
        st.stop()

    with st.spinner("Searching current web information and preparing results..."):
        try:
            data = research_mobile(model_name.strip())
        except Exception as exc:
            message = str(exc)

            if "429" in message:
                st.error(
                    "The Gemini API rate limit was reached. Please wait and try again."
                )
            elif "401" in message or "403" in message:
                st.error(
                    "The Gemini API key is invalid or does not have access."
                )
            else:
                st.error(f"Could not retrieve the information: {message}")

            st.stop()

    if not data.get("found", True):
        st.warning(
            "Sorry, I couldn't find reliable information for this model. "
            "Please check the model name and try again."
        )
        st.stop()

    model = data.get("model", {})
    display = data.get("display", {})
    performance = data.get("performance", {})
    camera = data.get("camera", {})
    battery = data.get("battery", {})
    features = data.get("features", {})

    st.success(f"Results for {safe_value(model.get('name'))}")

    st.header("📱 Overview")
    c1, c2, c3 = st.columns(3)
    with c1:
        show_field("Brand", model.get("brand"))
        show_field("Model", model.get("name"))
    with c2:
        show_field("Launch date", model.get("launch_date"))
        show_field("Operating system", model.get("operating_system"))
    with c3:
        show_field("Network", model.get("network"))
        show_field("SIM", model.get("sim"))

    st.header("🖥️ Display")
    c1, c2, c3 = st.columns(3)
    with c1:
        show_field("Size", display.get("size"))
        show_field("Type", display.get("type"))
    with c2:
        show_field("Resolution", display.get("resolution"))
        show_field("Refresh rate", display.get("refresh_rate"))
    with c3:
        show_field("Protection", display.get("protection"))

    st.header("⚡ Performance")
    c1, c2, c3 = st.columns(3)
    with c1:
        show_field("Processor", performance.get("processor"))
        show_field("GPU", performance.get("gpu"))
    with c2:
        show_field("RAM", performance.get("ram"))
    with c3:
        show_field("Storage", performance.get("storage"))

    st.header("📷 Camera")
    c1, c2 = st.columns(2)
    with c1:
        show_field("Main / rear", camera.get("rear_main"))
        show_field("Ultra-wide", camera.get("ultrawide"))
        show_field("Telephoto", camera.get("telephoto"))
    with c2:
        show_field("Front / selfie", camera.get("front"))
        show_field("Video", camera.get("video"))

    st.header("🔋 Battery")
    c1, c2, c3 = st.columns(3)
    with c1:
        show_field("Capacity", battery.get("capacity"))
    with c2:
        show_field("Charging", battery.get("charging"))
    with c3:
        show_field("Wireless charging", battery.get("wireless_charging"))

    st.header("🌐 Connectivity & Other Features")
    feature_pairs = [
        ("5G", "5g"),
        ("Fingerprint", "fingerprint"),
        ("Face unlock", "face_unlock"),
        ("Water resistance", "water_resistance"),
        ("Bluetooth", "bluetooth"),
        ("Wi-Fi", "wifi"),
        ("NFC", "nfc"),
        ("USB", "usb"),
        ("Weight", "weight"),
        ("Dimensions", "dimensions"),
        ("Colors", "colors"),
    ]

    for start in range(0, len(feature_pairs), 3):
        cols = st.columns(3)
        for col, (label, key) in zip(cols, feature_pairs[start:start + 3]):
            with col:
                show_field(label, features.get(key))

    st.header("💰 Pakistan Price")
    prices = data.get("prices_pakistan", [])

    if prices:
        price_table = []
        for item in prices:
            price_table.append(
                {
                    "Variant": safe_value(item.get("variant")),
                    "PTA status": safe_value(item.get("pta_status")),
                    "Price (PKR)": safe_value(item.get("price_pkr")),
                    "Type": safe_value(item.get("price_type")),
                    "Source": safe_value(item.get("source_name")),
                    "Date/context": safe_value(item.get("date_context")),
                }
            )

        st.dataframe(
            price_table,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Prices marked 'estimated' are not confirmed live prices. "
            "Verify price, PTA status, warranty, and stock with the retailer."
        )
    else:
        st.info("No reliable Pakistan price could be verified.")

    st.header("🏪 Lahore Availability")
    show_store_table(data.get("lahore", []))
    if not data.get("lahore"):
        st.caption(
            "Live stock availability could not be verified. "
            "Please contact the retailer before visiting."
        )

    st.header("🏪 Karachi Availability")
    show_store_table(data.get("karachi", []))
    if not data.get("karachi"):
        st.caption(
            "Live stock availability could not be verified. "
            "Please contact the retailer before visiting."
        )

    notes = data.get("notes", [])
    if notes:
        st.header("ℹ️ Notes")
        for note in notes:
            st.write(f"- {note}")

    sources = data.get("sources", [])
    if sources:
        st.header("🔗 Sources")
        for source in sources:
            name = safe_value(source.get("name"))
            url = safe_value(source.get("url"))
            if url != "Not available":
                st.markdown(f"- [{name}]({url})")
            else:
                st.write(f"- {name}")

st.divider()
st.caption(
    "Prices and store availability can change. Always verify current price, "
    "PTA status, warranty, and stock with the retailer before purchasing."
)
