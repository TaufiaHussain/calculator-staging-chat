from supabase import create_client, Client
import streamlit as st
from openai import OpenAI
from llm_router import handle_user_query
import pandas as pd
import math
import re
import datetime as dt

import hashlib, time  # time already used for filenames / ids
import io, json, hashlib, time, os
from datetime import datetime
from typing import Dict, Any, Callable

try:
    import numpy as np
except Exception:
    np = None

# Optional OCR for image input
try:
    import pytesseract
    from PIL import Image
    from pytesseract import TesseractNotFoundError

    # check if the Tesseract binary is actually available
    try:
        _ = pytesseract.get_tesseract_version()
        HAS_OCR = True
    except TesseractNotFoundError:
        HAS_OCR = False
except Exception:
    pytesseract = None
    Image = None
    HAS_OCR = False

# Which plan gets Tier-5 features (you can change this later)
TIER5_REQUIRED_PLAN = "pro"

# ------------------------------------------------------------
# PAGE
# ------------------------------------------------------------
st.set_page_config(page_title="Lab Solution Calculator", page_icon="🧪", layout="wide")

st.title("🧪 Versatile Lab Solution Calculator")
st.write("Dilutions, serials, plates, solids, % solutions, OD, master mixes, buffers, DMSO checks — all in one.")


@st.cache_resource
def get_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_ANON_KEY"]
    return create_client(url, key)


supabase = get_supabase()

# temporary: pretend this is the logged-in user for reagents
DEMO_USER_ID = "ff24c2dd-ea12-4f64-9a71-1d3c65df0647"


def load_reagents(user_id: str):
    data = (
        supabase.table("reagents")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )
    return data.data or []


if "fav_reagents" not in st.session_state:
    st.session_state["fav_reagents"] = []

reagents_db = load_reagents(DEMO_USER_ID)
for r in reagents_db:
    if r["name"] not in st.session_state["fav_reagents"]:
        st.session_state["fav_reagents"].append(r["name"])

# -------------------- TIER-4 HELPERS --------------------
# Protein extinction / MW from sequence
AA_MASS = {
    "A": 89.09, "R": 174.20, "N": 132.12, "D": 133.10, "C": 121.15,
    "E": 147.13, "Q": 146.15, "G": 75.07, "H": 155.16, "I": 131.17,
    "L": 131.17, "K": 146.19, "M": 149.21, "F": 165.19, "P": 115.13,
    "S": 105.09, "T": 119.12, "W": 204.23, "Y": 181.19, "V": 117.15
}
def protein_props_from_seq(seq: str):
    s = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", seq.upper())
    length = len(s)
    nW, nY, nC = s.count("W"), s.count("Y"), s.count("C") // 2  # assume half form cystine
    # ε at 280 nm (M^-1 cm^-1)
    epsilon = nW*5500 + nY*1490 + nC*125
    # MW (Da) — subtract water for peptide bonds: MW ≈ sum - (n-1)*18.015
    mw = sum(AA_MASS[a] for a in s) - max(0, length-1)*18.015
    return dict(length=length, nW=nW, nY=nY, nCystine=nC, epsilon=epsilon, mw=mw)

# Henderson–Hasselbalch (monoprotic)
def hh_pH(pKa, base_molar, acid_molar):
    if acid_molar <= 0: return None
    return pKa + math.log10(max(base_molar, 1e-12)/acid_molar)

# Osmolarity (sum of i*C) in Osm/L
VAN_T_HOFF = {"NaCl":2, "KCl":2, "CaCl2":3, "MgCl2":3, "Glucose":1, "Sucrose":1, "Urea":1}
def osmolarity(components):
    # components = list of dicts: {"name":str, "C_mM":float}
    Osm = 0.0
    for c in components:
        i = VAN_T_HOFF.get(c["name"], 1)
        Osm += i * (c["C_mM"]/1000.0)
    return Osm  # Osm/L

# Density table (g/mL) — approximate at 20–25°C
DENSITY = {
    "water": 0.998, "ethanol_100%": 0.789, "ethanol_70%": 0.867, "glycerol_100%": 1.261,
    "dmsO_100%": 1.100, "acetone_100%": 0.791
}

# Simple rules for incompatibility/precipitation flags
RULES = [
    ("phosphate buffer", "calcium", "Risk of Ca3(PO4)2 precipitation in high Ca2+."),
    ("dmso", "naoh", "Strong base may react with DMSO; avoid prolonged contact."),
    ("ethanol", "naoh", "Ethanol + strong base can form ethoxide; watch temperature."),
    ("triton", "protein quant", "Triton interferes with some dye-binding assays."),
    ("methanol", "pvp", "PVP can precipitate in high alcohol."),
]

def check_compatibility(text):
    t = text.lower()
    hits = []
    for a,b,msg in RULES:
        if a in t and b in t:
            hits.append(msg)
    return hits

# Linear regression (for spectrophotometry standard curves) if NumPy exists
def simple_linreg(x, y):
    if np is None or len(x) < 2: return None
    x = np.array(x, dtype=float); y = np.array(y, dtype=float)
    n = len(x)
    xm, ym = x.mean(), y.mean()
    ss_xy = ((x - xm) * (y - ym)).sum()
    ss_xx = ((x - xm)**2).sum()
    if ss_xx == 0: return None
    slope = ss_xy / ss_xx
    intercept = ym - slope*xm
    # R^2
    ss_yy = ((y - ym)**2).sum()
    yhat = slope*x + intercept
    ss_res = ((y - yhat)**2).sum()
    r2 = 1 - ss_res/ss_yy if ss_yy>0 else 0
    return dict(slope=slope, intercept=intercept, r2=r2)
# --------------------------------------------------------

# ------------------------------------------------------------
# Tier-5 helpers (hash, logging)
# ------------------------------------------------------------
def hash_api_key(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def save_run_to_cloud(
    mode: str,
    params: Dict[str, Any],
    result: Dict[str, Any],
    status: str = "done",
    error: str | None = None,
    upload_id: str | None = None,
    user_id: str | None = None,
):
    """Log a run in public.runs (best effort, no crash)."""
    try:
        supabase.table("runs").insert(
            {
                "user_id": user_id,
                "mode": mode,
                "params": params,
                "result": result,
                "status": status,
                "error": error,
                "source_upload": upload_id,
            }
        ).execute()
    except Exception:
        pass


def save_chat_msg(
    session_id: str | None,
    sender: str,
    content: str,
    tool_called: str | None = None,
    tool_input: Dict[str, Any] | None = None,
    tool_output: Dict[str, Any] | None = None,
):
    if not session_id:
        return
    try:
        supabase.table("chat_messages").insert(
            {
                "session_id": session_id,
                "sender": sender,
                "content": content,
                "tool_called": tool_called,
                "tool_input": tool_input,
                "tool_output": tool_output,
            }
        ).execute()
    except Exception:
        pass

def hash_api_key(raw: str) -> str:
    """Return a hex sha256 hash of the raw API key string."""
    return hashlib.sha256(raw.encode()).hexdigest()

# ------------------------------------------------------------
# 3) Auth helpers
# ------------------------------------------------------------
def login(email: str, password: str):
    """Sign in user with email+password."""
    return supabase.auth.sign_in_with_password({"email": email, "password": password})


def signup(email: str, password: str, full_name: str = ""):
    """Create new user (will trigger your SQL to create profile + subscription=free)."""
    return supabase.auth.sign_up(
        {
            "email": email,
            "password": password,
            "options": {"data": {"full_name": full_name}},
        }
    )


def logout():
    try:
        supabase.auth.sign_out()
    except Exception:
        pass
    # clear session stuff
    for key in ["user", "plan"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()


def get_subscription_plan(user_id: str) -> str:
    """
    Try to read the user's plan from public.subscriptions.
    If table is protected / empty / missing → fall back to 'free'.
    """
    try:
        resp = (
            supabase.table("subscriptions")
            .select("plan")
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        data = resp.data or []
        if len(data) > 0 and "plan" in data[0]:
            return data[0]["plan"]
        return "free"
    except Exception as e:
        # IMPORTANT: don't crash the app here
        st.info("Could not read subscription from Supabase → using FREE.")
        st.write(e)  # you can remove this later
        return "free"


# ------------------------------------------------------------
# 4) LOGIN GATE
# ------------------------------------------------------------
if "auth_session" not in st.session_state:
    st.session_state["auth_session"] = None
if "user" not in st.session_state:
    st.session_state["user"] = None

if st.session_state["user"] is None:
    st.title("🔐 Lab Solution Calculator (Login required)")

    tab_login, tab_signup = st.tabs(["Login", "Sign up"])

    with tab_login:
        lemail = st.text_input("Email", key="login_email")
        lpass = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            try:
                auth_res = login(lemail, lpass)
                st.session_state["auth_session"] = auth_res.session
                st.session_state["user"] = auth_res.user
                st.rerun()
            except Exception as e:
                st.error(f"Login failed: {e}")

    with tab_signup:
        semail = st.text_input("Email (for signup)", key="signup_email")
        sname = st.text_input("Full name", key="signup_name")
        spass = st.text_input("Password (min 6 chars)", type="password", key="signup_password")
        if st.button("Create account"):
            try:
                signup(semail, spass, sname)
                st.success("Account created. Now login from the Login tab.")
            except Exception as e:
                st.error(f"Signup failed: {e}")

    st.stop()  # 👈 do NOT show the app below

# ------------------------------------------------------------
# 5) USER IS LOGGED IN → CHECK SUBSCRIPTION
# ------------------------------------------------------------
user = st.session_state["user"]
plan = get_subscription_plan(user.id)

if plan != "pro":
    st.title("🧪 Lab Solution Calculator")
    st.warning("Your plan is **free**. This tool is for **Pro** users.")
    st.info("Ask admin to upgrade you in Supabase → public.subscriptions, or connect Stripe later.")
    st.stop()

# ------------------------------------------------------------
# optional PDF
# ------------------------------------------------------------
try:
    from fpdf import FPDF

    HAS_FPDF = True
except Exception:
    HAS_FPDF = False


def make_pdf_report(title: str, lines: list[str]) -> bytes | None:
    """Small helper to build a PDF from lines. Needs fpdf."""
    if not HAS_FPDF:
        return None
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, txt=title, ln=1)
    pdf.ln(3)
    for ln in lines:
        pdf.multi_cell(0, 6, txt=ln)
    return pdf.output(dest="S").encode("latin-1")


# ------------------------------------------------------------
# SIDEBAR: user profile / dark mode / presets
# ------------------------------------------------------------
if "fav_reagents" not in st.session_state:
    st.session_state["fav_reagents"] = []
if "username" not in st.session_state:
    st.session_state["username"] = ""

dark_mode = st.sidebar.checkbox("🌙 Dark mode", value=False)

if dark_mode:
    st.markdown(
        """
        <style>
        body, .stApp {
            background: #0f172a;
            color: #ffffff;
        }
        .stButton>button {
            background: #1f2937;
            color: #fff;
        }
        .stDataFrame {
            background: #0f172a;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# lab presets
preset = st.sidebar.selectbox(
    "Lab preset",
    [
        "Custom",
        "Cell culture (0.1% DMSO, 300 µl)",
        "Chemistry (no vehicle, 1000 µl)",
        "qPCR / assay (20 µl)",
    ],
)

st.sidebar.header("Global settings")

# defaults from preset
_default_well = 300.0
_default_max_vehicle = 0.1
_default_vehicle_type = "Aqueous / none"
_default_stock_vehicle_percent = 100.0

if preset == "Cell culture (0.1% DMSO, 300 µl)":
    _default_well = 300.0
    _default_max_vehicle = 0.1
    _default_vehicle_type = "DMSO"
elif preset == "Chemistry (no vehicle, 1000 µl)":
    _default_well = 1000.0
    _default_max_vehicle = 0.0
    _default_vehicle_type = "Aqueous / none"
elif preset == "qPCR / assay (20 µl)":
    _default_well = 20.0
    _default_max_vehicle = 0.5
    _default_vehicle_type = "Aqueous / none"

# actual inputs
well_volume = st.sidebar.number_input("Default well / final volume (µl)", value=_default_well, min_value=1.0)

max_vehicle = st.sidebar.number_input(
    "Max allowed DMSO/EtOH in final (%)",
    value=_default_max_vehicle,
    min_value=0.0,
    step=0.05,
    help="Typical cell culture limit is 0.1–0.5 %.",
)

vehicle_type = st.sidebar.selectbox(
    "Stock vehicle",
    ["Aqueous / none", "DMSO", "EtOH"],
    index=["Aqueous / none", "DMSO", "EtOH"].index(_default_vehicle_type),
    help="Select the solvent in which your STOCK is dissolved.",
)

stock_vehicle_percent = st.sidebar.number_input(
    "Stock vehicle % (e.g. 100 for pure DMSO, 50 for 1:1 DMSO:water)",
    value=_default_stock_vehicle_percent,
    min_value=0.0,
    max_value=100.0,
    step=5.0,
)

vehicle_frac = 0.0
if vehicle_type != "Aqueous / none" and stock_vehicle_percent > 0:
    vehicle_frac = stock_vehicle_percent / 100.0

# user profile / favorites
with st.sidebar.expander("👤 User profile / favorites", expanded=False):
    username = st.text_input("Your name", value=st.session_state["username"])
    st.session_state["username"] = username
    if st.session_state["fav_reagents"]:
        st.write("⭐ Saved reagents:")
        for r in st.session_state["fav_reagents"]:
            st.write("- ", r)
    else:
        st.write("No saved reagents yet.")

with st.sidebar:
    user = st.session_state.get("user")
    if user:
        st.markdown(f"**Logged in as:** {user.email}")
        if st.button("Logout"):
            logout()
    else:
        st.info("Please log in to use Pro tools.")

# ------------------------------------------------------------
# MAIN MODE SELECTOR
# ------------------------------------------------------------
mode = st.selectbox(
    "Select calculator mode:",
    [
        "Single dilution (C1V1 = C2V2)",
        "Serial dilutions",
        "Experiment series (plate-like)",
        "From solid (mg → solution)",
        "Unit converter (mg/mL ↔ mM)",
        "% solutions (w/v, v/v)",
        "Molarity from mass & volume",
        "OD / culture dilution",
        "Master mix / qPCR mix",
        "Make X× stock from current stock",
        "Acid / base dilution (common reagents)",
        "Buffer helper (PBS / TBS / Tris)",
        "Beer–Lambert / A280",
        "Cell seeding calculator",
        "Plate DMSO cap checker",
        "Aliquot splitter",
        "Storage / stability helper",
        "Protein extinction / MW from sequence",
        "pH & buffer capacity",
        "Cell culture media designer",
        "Primer / probe concentration helper",
        "Inventory tracker (Pro)",
        "Reagent stability predictor",
        "Dilution series visualizer",
        "Notebook generator (PDF/MD)",
        "Osmolarity calculator",
        "Spectrophotometry toolbox",
        "Solution density converter",
        "Reagent compatibility checker",
    ],
)

# share calculation (stores mode in URL)
if st.button("🔗 Make this URL shareable for this mode"):
    st.experimental_set_query_params(mode=mode)
    st.success("Query params set. Copy the URL from your browser and share it.")

# ======================================================================
# 1) SINGLE DILUTION
# ======================================================================
if mode == "Single dilution (C1V1 = C2V2)":
    st.subheader("Single dilution")

    col1, col2 = st.columns(2)
    with col1:
        stock_conc = st.number_input("Stock concentration", value=25.0, min_value=0.000001)
        stock_unit = st.selectbox("Stock unit", ["mM", "µM"])
    with col2:
        target_conc = st.number_input("Target concentration", value=4.0, min_value=0.000001)
        target_unit = st.selectbox("Target unit", ["mM", "µM"])

    show_steps = st.checkbox("Show protocol-style steps", value=True)

    if stock_unit != target_unit:
        st.error("For now, keep units the same (mM→mM or µM→µM).")
    else:
        v1_ul = (target_conc * well_volume) / stock_conc
        solvent_ul = well_volume - v1_ul
        if solvent_ul < 0:
            solvent_ul = 0.0
        vehicle_percent = (v1_ul * vehicle_frac / well_volume) * 100

        st.markdown("### Result")
        st.write(f"- Pipette **{v1_ul:.2f} µl** from stock")
        st.write(f"- Add solvent / medium **{solvent_ul:.2f} µl** to reach **{well_volume:.0f} µl**")
        st.write(f"- Final vehicle (DMSO/EtOH): **{vehicle_percent:.4f} %**")

        if vehicle_percent > max_vehicle:
            st.warning(
                f"Vehicle {vehicle_percent:.4f}% > allowed {max_vehicle:.2f}%. "
                "Make a more dilute stock OR increase final volume."
            )

        min_pip = 1.0
        if v1_ul < min_pip:
            c_intermediate = (target_conc * well_volume) / min_pip
            st.warning(
                f"Volume from stock is very small ({v1_ul:.3f} µl). "
                f"👉 Make an intermediate stock ≈ {c_intermediate:.3f} {target_unit} and repeat."
            )

        if show_steps:
            st.markdown("#### Steps")
            st.markdown(f"1. Label a tube with target conc: **{target_conc} {target_unit}**.")
            st.markdown(f"2. Pipette **{v1_ul:.2f} µl** of the stock solution into the tube.")
            st.markdown(f"3. Add **{solvent_ul:.2f} µl** of medium / buffer.")
            st.markdown("4. Mix gently. Protect from light if compound is light-sensitive.")
            st.markdown("5. Use immediately or aliquot / store as protocol allows.")

        # PDF
        if HAS_FPDF:
            if st.button("📄 Export this as PDF"):
                lines = [
                    "Mode: Single dilution",
                    f"Stock: {stock_conc} {stock_unit}",
                    f"Target: {target_conc} {target_unit}",
                    f"Final volume: {well_volume} µl",
                    f"Take from stock: {v1_ul:.2f} µl",
                    f"Add solvent: {solvent_ul:.2f} µl",
                    f"Vehicle: {vehicle_percent:.4f} %",
                ]
                pdf_bytes = make_pdf_report("Single dilution report", lines)
                st.download_button("⬇ Download PDF", data=pdf_bytes, file_name="single_dilution.pdf")
        else:
            st.info("Install `fpdf` to enable PDF export: `pip install fpdf`")

# ======================================================================
# 2) SERIAL DILUTIONS
# ======================================================================
elif mode == "Serial dilutions":
    st.subheader("Serial dilutions")

    start_conc = st.number_input("Start concentration (mM)", value=25.0, min_value=0.000001)
    n_steps = st.number_input("Number of dilutions", value=5, min_value=1, step=1)
    dil_factor = st.number_input("Dilution factor (e.g. 2 for 1:2)", value=2.0, min_value=1.0001)
    final_vol_each = st.number_input("Final volume for each tube (µl)", value=100.0, min_value=5.0)

    rows = []
    current_conc = start_conc
    min_pip = 1.0
    for i in range(int(n_steps)):
        next_conc = current_conc / dil_factor
        v1_ul = (next_conc * final_vol_each) / current_conc
        solvent_ul = final_vol_each - v1_ul
        vehicle_percent = (v1_ul * vehicle_frac / final_vol_each) * 100

        warning_flag = ""
        if v1_ul < min_pip:
            warning_flag = "<1 µl → make intermediate"

        rows.append(
            {
                "step": i + 1,
                "from (mM)": round(current_conc, 6),
                "to (mM)": round(next_conc, 6),
                "take from prev (µl)": round(v1_ul, 3),
                "add solvent (µl)": round(solvent_ul, 3),
                "vehicle %": round(vehicle_percent, 5),
                "note": warning_flag,
            }
        )

        current_conc = next_conc

    df = pd.DataFrame(rows)
    st.write("### Dilution plan")
    st.dataframe(df)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download CSV", data=csv, file_name="serial_dilutions.csv")

    if HAS_FPDF:
        if st.button("📄 Export as PDF"):
            lines = ["Serial dilution plan:"]
            for r in rows:
                lines.append(str(r))
            pdf_bytes = make_pdf_report("Serial dilutions", lines)
            st.download_button("⬇ Download PDF", data=pdf_bytes, file_name="serial_dilutions.pdf")

# ======================================================================
# 3) EXPERIMENT SERIES (PLATE-LIKE)
# ======================================================================
elif mode == "Experiment series (plate-like)":
    st.subheader("Experiment series (fixed final volume)")

    st.write("Enter final concentrations (µM), separated by commas, e.g. `0.01,0.1,1,3,10`")
    conc_txt = st.text_input("Final concentrations (µM)", value="0.01,0.1,1,3,10")
    stock_conc_uM = st.number_input("Stock concentration (µM)", value=10000.0, min_value=0.0001)

    reps = st.number_input("Replicates per concentration (wells)", value=3, min_value=1, step=1)
    overfill = st.number_input(
        "Overfill factor (1.0 = exact, 1.1 = +10%)", value=1.1, min_value=1.0, step=0.05
    )

    concs = [float(x.strip()) for x in conc_txt.split(",") if x.strip()]
    table = []
    for c in concs:
        v1_ul = (c * well_volume) / stock_conc_uM
        solvent_ul = well_volume - v1_ul
        vehicle_percent = (v1_ul * vehicle_frac / well_volume) * 100
        total_vol_ul = (v1_ul + solvent_ul) * reps * overfill

        table.append(
            {
                "final conc (µM)": c,
                "add stock (µl) / well": round(v1_ul, 3),
                "add medium (µl) / well": round(solvent_ul, 3),
                "vehicle %": round(vehicle_percent, 5),
                "OK?": "⚠ > limit" if vehicle_percent > max_vehicle else "✅",
                "total vol to prepare (µl)": round(total_vol_ul, 1),
            }
        )

    df = pd.DataFrame(table)
    st.dataframe(df)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download dilution plan as CSV",
        csv,
        "dilution_plan.csv",
        "text/csv",
    )

    if HAS_FPDF:
        if st.button("📄 Export as PDF"):
            lines = ["Experiment series plan:"]
            for row in table:
                lines.append(str(row))
            pdf_bytes = make_pdf_report("Experiment series", lines)
            st.download_button("⬇ Download PDF", data=pdf_bytes, file_name="experiment_series.pdf")

# ======================================================================
# 4) FROM SOLID (MG → SOLUTION)
# ======================================================================
elif mode == "From solid (mg → solution)":
    st.subheader("From solid (mg → solution)")
    st.write("Use this when you only know how many mg you bought and you want a certain µM/mM in a given volume.")

    compound = st.selectbox(
        "Choose compound (optional)",
        [
            "-- custom --",
            "Retinal (284.44)",
            "AMPA (192.17)",
            "Forskolin (410.5)",
            "Retinoic acid (300.44)",
            "GABA (103.12)",
        ],
    )

    light_sensitive_words = ["retinal", "retinoic", "rhodamine", "fitc"]

    if compound == "Retinal (284.44)":
        default_mw = 284.44
        compound_name = "retinal"
    elif compound == "AMPA (192.17)":
        default_mw = 192.17
        compound_name = "ampa"
    elif compound == "Forskolin (410.5)":
        default_mw = 410.5
        compound_name = "forskolin"
    elif compound == "Retinoic acid (300.44)":
        default_mw = 300.44
        compound_name = "retinoic acid"
    elif compound == "GABA (103.12)":
        default_mw = 103.12
        compound_name = "gaba"
    else:
        default_mw = 284.44  # safe default
        compound_name = st.text_input("Compound name (for notes / warnings)", "")

    mass_mg = st.number_input("Mass (mg)", value=50.0, min_value=0.0001)
    mw = st.number_input("Molecular weight (g/mol)", value=default_mw, min_value=1.0)
    target_unit = st.selectbox("Target concentration unit", ["µM", "mM"])
    target_conc = st.number_input("Target concentration", value=100.0, min_value=0.0001)
    final_vol_ml = st.number_input("Final volume to prepare (mL)", value=20.0, min_value=0.1)

    if target_unit == "µM":
        C = target_conc * 1e-6  # mol/L
    else:
        C = target_conc * 1e-3  # mol/L

    V_L = final_vol_ml / 1000.0
    m_needed_g = C * V_L * mw
    m_needed_mg = m_needed_g * 1000

    st.markdown(
        f"**To make {final_vol_ml:.1f} mL at {target_conc} {target_unit}, you must weigh ≈ {m_needed_mg:.3f} mg.**"
    )

    # if I dissolve everything
    mass_g = mass_mg / 1000.0
    n_mol = mass_g / mw
    stock_if_1ml_mM = (n_mol / 0.001) * 1000  # mol/L → mM
    stock_if_2ml_mM = (n_mol / 0.002) * 1000

    st.write("If you dissolve ALL your powder:")
    st.write(f"- in **1.0 mL** → **{stock_if_1ml_mM:.1f} mM** stock")
    st.write(f"- in **2.0 mL** → **{stock_if_2ml_mM:.1f} mM** stock")

    name_to_check = (compound_name or "").lower()
    if any(word in name_to_check for word in light_sensitive_words):
        st.warning(
            "This compound looks light-sensitive. Protect from light (amber tube / foil), "
            "use dry EtOH or DMSO, aliquot, store cold."
        )

    # save to favorites
    if st.button("⭐ Save this reagent to my favorites"):
        if compound_name:
            # save in session
            if compound_name not in st.session_state["fav_reagents"]:
                st.session_state["fav_reagents"].append(compound_name)

            # save in Supabase
            supabase.table("reagents").insert(
                {
                    "user_id": DEMO_USER_ID,
                    "name": compound_name,
                    "mw": mw,
                    "note": "from app",
                }
            ).execute()

            st.success(f"Saved '{compound_name}' to Supabase + session.")
        else:
            st.warning("Give the compound a name first.")

    if HAS_FPDF:
        if st.button("📄 Export this as PDF"):
            lines = [
                f"Compound: {compound_name or compound}",
                f"Mass available: {mass_mg} mg",
                f"MW: {mw} g/mol",
                f"Target: {target_conc} {target_unit} in {final_vol_ml} mL",
                f"Mass needed: {m_needed_mg:.3f} mg",
                f"Stock if all dissolved in 1 mL: {stock_if_1ml_mM:.1f} mM",
                f"Stock if all dissolved in 2 mL: {stock_if_2ml_mM:.1f} mM",
            ]
            pdf_bytes = make_pdf_report("Solid → solution report", lines)
            st.download_button("⬇ Download PDF", data=pdf_bytes, file_name="solid_to_solution.pdf")

# ======================================================================
# 5) UNIT CONVERTER
# ======================================================================
elif mode == "Unit converter (mg/mL ↔ mM)":
    st.subheader("Unit converter (mg/mL ↔ mM)")

    mw = st.number_input("Molecular weight (g/mol)", value=284.44, min_value=1.0)
    direction = st.radio("Convert", ["mg/mL → mM", "mM → mg/mL"])

    if direction == "mg/mL → mM":
        mgml = st.number_input("Concentration (mg/mL)", value=1.0, min_value=0.0)
        mM = (mgml * 1000.0) / mw
        st.success(f"{mgml} mg/mL  →  {mM:.3f} mM")
    else:
        mM = st.number_input("Concentration (mM)", value=1.0, min_value=0.0)
        mgml = (mM * mw) / 1000.0
        st.success(f"{mM} mM  →  {mgml:.3f} mg/mL")

# ======================================================================
# 6) % SOLUTIONS
# ======================================================================
elif mode == "% solutions (w/v, v/v)":
    st.subheader("% solutions (w/v, v/v)")

    percent_type = st.radio("Type", ["w/v (g per 100 mL)", "v/v (mL per 100 mL)"])

    final_vol_ml = st.number_input("Final volume (mL)", value=100.0, min_value=1.0)
    percent = st.number_input("Percent (%)", value=2.0, min_value=0.0)

    if percent_type == "w/v (g per 100 mL)":
        grams_needed = (percent / 100.0) * final_vol_ml
        st.success(f"To make {percent}% w/v, weigh **{grams_needed:.3f} g** and bring volume to {final_vol_ml:.1f} mL.")
    else:
        ml_needed = (percent / 100.0) * final_vol_ml
        st.success(
            f"To make {percent}% v/v, measure **{ml_needed:.3f} mL** of solute and add solvent to {final_vol_ml:.1f} mL."
        )

# ======================================================================
# 7) MOLARITY FROM MASS & VOLUME
# ======================================================================
elif mode == "Molarity from mass & volume":
    st.subheader("Molarity from mass & volume")
    st.write("Example: I dissolved 12 mg in 10 mL, what is the molarity?")

    mass_mg = st.number_input("Mass dissolved (mg)", value=12.0, min_value=0.0)
    mw = st.number_input("Molecular weight (g/mol)", value=284.44, min_value=1.0)
    vol_ml = st.number_input("Final volume (mL)", value=10.0, min_value=0.01)

    mass_g = mass_mg / 1000.0
    vol_L = vol_ml / 1000.0
    if vol_L > 0:
        moles = mass_g / mw
        molarity = moles / vol_L
        st.success(f"Molarity = **{molarity:.4f} M** ({molarity*1000:.2f} mM)")
    else:
        st.error("Volume must be > 0")

# ======================================================================
# 8) OD / CULTURE DILUTION
# ======================================================================
elif mode == "OD / culture dilution":
    st.subheader("OD / culture dilution")
    st.write("C1V1 = C2V2, but for cultures.")

    od_start = st.number_input("Starting OD / cell density (C1)", value=1.2, min_value=0.0001)
    od_target = st.number_input("Target OD (C2)", value=0.1, min_value=0.0001)
    final_vol_ml = st.number_input("Final volume to prepare (mL)", value=10.0, min_value=0.1)

    v1_ml = (od_target * final_vol_ml) / od_start
    diluent_ml = final_vol_ml - v1_ml

    st.write(f"- Take **{v1_ml:.2f} mL** of culture")
    st.write(f"- Add **{diluent_ml:.2f} mL** of medium to reach **{final_vol_ml:.2f} mL** at OD {od_target}")

# ======================================================================
# 9) MASTER MIX / qPCR MIX
# ======================================================================
elif mode == "Master mix / qPCR mix":
    st.subheader("Master mix / qPCR mix")

    n_rxn = st.number_input("Number of reactions", value=10, min_value=1, step=1)
    rxn_vol_ul = st.number_input("Reaction volume (µl)", value=20.0, min_value=5.0)
    overfill = st.number_input(
        "Overfill factor (1.0 = exact, 1.1 = +10%)", value=1.1, min_value=1.0, step=0.05
    )

    st.write("Specify components in µL per reaction:")
    col1, col2, col3 = st.columns(3)
    with col1:
        buf = st.number_input("Buffer / Master mix (µl)", value=10.0, min_value=0.0)
        dntp = st.number_input("dNTP / MgCl2 (µl)", value=0.0, min_value=0.0)
    with col2:
        primer_f = st.number_input("Primer F (µl)", value=0.5, min_value=0.0)
        primer_r = st.number_input("Primer R (µl)", value=0.5, min_value=0.0)
    with col3:
        template = st.number_input("Template (µl)", value=1.0, min_value=0.0)
        polymerase = st.number_input("Polymerase / enzyme (µl)", value=0.2, min_value=0.0)

    per_rxn_sum = buf + dntp + primer_f + primer_r + template + polymerase
    other_needed = rxn_vol_ul - per_rxn_sum
    if other_needed < 0:
        st.error("Sum of components exceeds reaction volume — reduce some components.")
    total_rxn = n_rxn * overfill

    st.markdown("### Total mix to prepare")
    st.write(f"- Buffer / Master mix: **{buf * total_rxn:.2f} µl**")
    st.write(f"- dNTP / MgCl2: **{dntp * total_rxn:.2f} µl**")
    st.write(f"- Primer F: **{primer_f * total_rxn:.2f} µl**")
    st.write(f"- Primer R: **{primer_r * total_rxn:.2f} µl**")
    st.write(f"- Polymerase: **{polymerase * total_rxn:.2f} µl**")
    st.write(f"- Template (add separately if different): **{template * n_rxn:.2f} µl**")
    if other_needed > 0:
        st.write(f"- Nuclease-free water: **{other_needed * total_rxn:.2f} µl**")

# ======================================================================
# 10) MAKE X× STOCK
# ======================================================================
elif mode == "Make X× stock from current stock":
    st.subheader("Make X× stock from current stock")

    current_conc = st.number_input("Current concentration (e.g. 1×)", value=1.0, min_value=0.0001)
    desired_mult = st.number_input("Desired stock multiple (e.g. 10 for 10×)", value=10.0, min_value=1.0)
    final_vol_ml = st.number_input("Final stock volume to make (mL)", value=50.0, min_value=1.0)

    V1 = final_vol_ml / desired_mult
    solvent_ml = final_vol_ml - V1

    st.write(f"- Take **{V1:.2f} mL** of your current solution")
    st.write(f"- Add **{solvent_ml:.2f} mL** solvent to get **{final_vol_ml:.2f} mL** of **{desired_mult:.0f}×**")

# ======================================================================
# 11) ACID / BASE DILUTION
# ======================================================================
elif mode == "Acid / base dilution (common reagents)":
    st.subheader("Acid / base dilution (common reagents)")
    st.write("Compute volume of concentrated reagent (HCl, H₂SO₄, NH₃) to make a given molarity & volume.")

    reagents = {
        "HCl 37%": {"density": 1.19, "purity": 0.37, "mw": 36.46},
        "H2SO4 98%": {"density": 1.84, "purity": 0.98, "mw": 98.08},
        "NH3 25%": {"density": 0.91, "purity": 0.25, "mw": 17.03},
    }

    reagent_name = st.selectbox("Reagent", list(reagents.keys()))
    target_m = st.number_input("Target molarity (M)", value=1.0, min_value=0.0001)
    final_vol_L = st.number_input("Final volume (L)", value=1.0, min_value=0.01)

    r = reagents[reagent_name]
    moles_needed = target_m * final_vol_L
    mass_pure = moles_needed * r["mw"]
    mass_conc = mass_pure / r["purity"]
    vol_conc_L = mass_conc / r["density"]
    vol_conc_ml = vol_conc_L * 1000

    st.success(
        f"For {target_m} M {reagent_name} in {final_vol_L} L:\n"
        f"- Weigh/measure **{vol_conc_ml:.1f} mL** of concentrated {reagent_name}\n"
        f"- Add to water and bring to volume."
    )
    st.info("Always add acid to water, not water to acid.")

# ======================================================================
# 12) BUFFER HELPER
# ======================================================================
elif mode == "Buffer helper (PBS / TBS / Tris)":
    st.subheader("Buffer helper")

    buffer_type = st.selectbox("Buffer", ["PBS 1× (1 L)", "PBS 10× (1 L)", "TBS 1× (1 L)", "Tris 1 M (pH 8.0, 1 L)"])

    if buffer_type == "PBS 1× (1 L)":
        st.write("**PBS 1× (pH 7.4) for 1 L**")
        st.write("- NaCl: 8.0 g")
        st.write("- KCl: 0.2 g")
        st.write("- Na2HPO4 (anhydrous): 1.44 g")
        st.write("- KH2PO4: 0.24 g")
        st.write("- Dissolve in ~800 mL, adjust pH, bring to 1 L.")
    elif buffer_type == "PBS 10× (1 L)":
        st.write("**PBS 10× (pH 7.4) for 1 L**")
        st.write("- NaCl: 80 g")
        st.write("- KCl: 2 g")
        st.write("- Na2HPO4: 14.4 g")
        st.write("- KH2PO4: 2.4 g")
        st.write("- Dissolve, adjust, bring to 1 L.")
    elif buffer_type == "TBS 1× (1 L)":
        st.write("**TBS 1× for 1 L**")
        st.write("- NaCl: 8.0 g")
        st.write("- Tris base: 3.0 g")
        st.write("- Adjust pH to 7.4–7.6 with HCl, bring to 1 L.")
    else:
        st.write("**Tris 1 M pH 8.0 (1 L)**")
        st.write("- Tris base (MW 121.14): 121.14 g")
        st.write("- Dissolve ~800 mL, adjust pH with HCl, bring to 1 L.")

# ======================================================================
# 13) BEER–LAMBERT
# ======================================================================
elif mode == "Beer–Lambert / A280":
    st.subheader("Beer–Lambert / A280")

    absorbance = st.number_input("Absorbance (A)", value=0.5, min_value=0.0)
    epsilon = st.number_input("Extinction coefficient (M⁻¹ cm⁻¹)", value=50000.0, min_value=1.0)
    pathlength = st.number_input("Pathlength (cm)", value=1.0, min_value=0.01)

    if epsilon > 0 and pathlength > 0:
        conc_M = absorbance / (epsilon * pathlength)
        st.success(f"Concentration = {conc_M:.6f} M  ({conc_M*1000:.3f} mM)")
    else:
        st.error("Epsilon and pathlength must be > 0")

# ======================================================================
# 14) CELL SEEDING
# ======================================================================
elif mode == "Cell seeding calculator":
    st.subheader("Cell seeding calculator")

    stock_density = st.number_input("Current cell suspension (cells/mL)", value=1_500_000, min_value=1)
    target_density = st.number_input("Target cells per well/dish", value=200_000, min_value=1)
    final_volume_ml = st.number_input("Final volume per well/dish (mL)", value=2.0, min_value=0.1)

    vol_cells_ml = target_density / stock_density
    vol_medium_ml = final_volume_ml - vol_cells_ml

    st.write(f"- Take **{vol_cells_ml:.3f} mL** of cell suspension")
    st.write(f"- Add **{vol_medium_ml:.3f} mL** of medium to reach {final_volume_ml:.2f} mL with {target_density} cells")

# ======================================================================
# 15) PLATE DMSO CAP CHECKER
# ======================================================================
elif mode == "Plate DMSO cap checker":
    st.subheader("Plate DMSO cap checker")
    st.write("Enter final concentrations (µM) exactly like in the plate-like mode. We'll flag any wells > DMSO limit.")

    conc_txt = st.text_input("Final concentrations (µM)", value="0.01,0.1,1,3,10")
    stock_conc_uM = st.number_input("Stock concentration (µM)", value=10000.0, min_value=0.0001)
    dmso_cap = st.number_input("DMSO cap (%)", value=max_vehicle, min_value=0.0, step=0.05)

    concs = [float(x.strip()) for x in conc_txt.split(",") if x.strip()]
    rows = []
    for c in concs:
        v1_ul = (c * well_volume) / stock_conc_uM
        dmso_percent = (v1_ul * vehicle_frac / well_volume) * 100
        rows.append(
            {
                "final conc (µM)": c,
                "stock vol (µl)": round(v1_ul, 3),
                "DMSO / EtOH %": round(dmso_percent, 5),
                "OK?": "✅" if dmso_percent <= dmso_cap else "⚠ EXCEEDS",
            }
        )

    df = pd.DataFrame(rows)
    st.dataframe(df)

# ======================================================================
# 16) ALIQUOT SPLITTER
# ======================================================================
elif mode == "Aliquot splitter":
    st.subheader("Aliquot splitter")

    total_vol_ml = st.number_input("Total volume you have (mL)", value=2.0, min_value=0.01)
    aliquot_vol_ml = st.number_input("Aliquot size (mL)", value=0.1, min_value=0.001)
    dead_vol_ml = st.number_input("Keep dead volume (mL)", value=0.0, min_value=0.0)

    usable_vol_ml = total_vol_ml - dead_vol_ml
    if usable_vol_ml <= 0:
        st.error("Dead volume is ≥ total volume.")
    else:
        n_aliquots = math.floor(usable_vol_ml / aliquot_vol_ml)
        leftover = usable_vol_ml - n_aliquots * aliquot_vol_ml
        st.write(f"- You can make **{n_aliquots} aliquots** of {aliquot_vol_ml} mL")
        st.write(f"- Leftover (not aliquoted): **{leftover:.3f} mL**")
        if dead_vol_ml > 0:
            st.info(f"{dead_vol_ml} mL reserved as dead volume.")

# ======================================================================
# 17) PROTEIN EXTINCTION / MW FROM SEQUENCE
# ======================================================================
elif mode== "Protein extinction / MW from sequence":
    st.subheader("Protein extinction / MW from sequence")
    seq = st.text_area("Paste amino acid sequence (one letter)", height=160,
                       placeholder="MAGRSGT... (letters: ACDEFGHIKLMNPQRSTVWY)")
    if seq.strip():
        props = protein_props_from_seq(seq)
        st.write(f"**Length:** {props['length']} aa")
        st.write(f"**Tryptophan:** {props['nW']}, **Tyrosine:** {props['nY']}, **Cystine (pairs):** {props['nCystine']}")
        st.write(f"**ε(280 nm):** {props['epsilon']:.0f} M⁻¹·cm⁻¹")
        st.write(f"**Molecular weight:** {props['mw']:.1f} Da")
        A = st.number_input("Measured A280", value=0.5, min_value=0.0)
        path = st.number_input("Pathlength (cm)", value=1.0, min_value=0.01)
        if props['epsilon']>0 and path>0:
            conc_M = A / (props['epsilon']*path)
            st.success(f"Estimated concentration: **{conc_M*1e6:.2f} µM**")

# ======================================================================
# 18) pH & BUFFER CAPACITY
# ======================================================================
elif mode == "pH & buffer capacity":
    st.subheader("pH & buffer capacity (monoprotic)")
    pKa = st.number_input("pKa", value=7.21, help="e.g., Tris ~8.1, Acetate ~4.76, Phosphate pKa2 ~7.21")
    acid = st.number_input("Acid form [HA] (mM)", value=50.0, min_value=0.0)
    base = st.number_input("Base form [A-] (mM)", value=50.0, min_value=0.0)
    pH = hh_pH(pKa, base/1000.0, acid/1000.0)
    if pH is not None:
        st.success(f"Predicted pH: **{pH:.2f}**")
        # crude buffer capacity around pH = pKa: β ≈ 2.3·C_total·(Ka·[H+])/([H+]+Ka)^2
        Ka = 10**(-pKa)
        H = 10**(-pH)
        Ctot = (acid+base)/1000.0  # M
        beta = 2.303 * Ctot * (Ka*H)/((H+Ka)**2)
        st.write(f"Approx. buffer capacity near this pH: **{beta:.4f} mol·L⁻¹·pH⁻¹** (higher is more resistant to pH change)")

# ======================================================================
# 19) CELL CULTURE MEDIA DESIGNER
# ======================================================================
elif mode == "Cell culture media designer":
    st.subheader("Cell culture media designer")
    cell = st.selectbox("Cell type", ["HEK293", "CHO", "Drosophila S2", "Primary neurons", "Custom"])
    serum = st.slider("Serum (FBS, %)", 0, 20, 10)
    antibiotics = st.checkbox("Pen/Strep (1×)", value=True)
    glutamine = st.checkbox("L-Glutamine (2 mM)", value=True)
    notes = ""
    if cell == "HEK293":
        base = "DMEM high glucose"
        notes = "Optional: 1 mM sodium pyruvate."
    elif cell == "CHO":
        base = "Ham's F12 or DMEM/F12"
    elif cell == "Drosophila S2":
        base = "Schneider’s Drosophila Medium"
        serum = max(serum, 10)
    elif cell == "Primary neurons":
        base = "Neurobasal + B27"
        serum = 0
        glutamine = True
    else:
        base = st.text_input("Base medium", "DMEM/F12")
    st.markdown("### Recipe")
    st.write(f"- Base medium: **{base}**")
    st.write(f"- Serum: **{serum}%**")
    if antibiotics: st.write("- Pen/Strep: **1×**")
    if glutamine: st.write("- L-Glutamine: **2 mM**")
    if notes: st.info(notes)

# ======================================================================
# 20) PRIMER / PROBE CONCENTRATION HELPER
# ======================================================================
elif mode == "Primer / probe concentration helper":
    st.subheader("Primer / probe concentration helper")
    stock_unit = st.selectbox("Stock unit", ["µM", "ng/µL"])
    if stock_unit == "µM":
        stock = st.number_input("Stock (µM)", value=100.0, min_value=0.0)
        target = st.number_input("Working (µM)", value=0.5, min_value=0.0)
        final_vol = st.number_input("Final volume (µL)", value=20.0, min_value=1.0)
        v_stock = (target * final_vol) / stock if stock>0 else 0
        st.success(f"Add **{v_stock:.2f} µL** stock + **{final_vol - v_stock:.2f} µL** buffer to reach {target} µM.")
    else:
        mw_bp = st.number_input("Oligo MW (g/mol) or use approx 330×nt", value=330.0*20, min_value=1.0)
        stock_ng = st.number_input("Stock (ng/µL)", value=100.0, min_value=0.0)
        # Convert ng/µL → µM: µM = (ng/µL) / (MW g/mol) * (10^3 µL/mL) * (10^6 µmol/mol)
        stock_uM = (stock_ng/1e9) / (mw_bp) * 1e6 * 1e3
        st.write(f"Approx stock: **{stock_uM:.2f} µM**")
        target = st.number_input("Working (µM)", value=0.5, min_value=0.0)
        final_vol = st.number_input("Final volume (µL)", value=20.0, min_value=1.0)
        v_stock = (target * final_vol) / stock_uM if stock_uM>0 else 0
        st.success(f"Add **{v_stock:.2f} µL** stock + **{final_vol - v_stock:.2f} µL** buffer to reach {target} µM.")

# ======================================================================
# 21) INVENTORY TRACKER (Pro)
# ======================================================================
elif mode == "Inventory tracker (Pro)":
    st.subheader("Inventory tracker (Pro)")
    st.caption("Simple local tracker. If Supabase is configured, you can map this to a table later.")
    if "inventory" not in st.session_state:
        st.session_state["inventory"] = []
    with st.form("add_item"):
        c1,c2,c3 = st.columns(3)
        with c1:
            name = st.text_input("Name", "")
            conc = st.text_input("Concentration (e.g., 10 mM)", "")
        with c2:
            loc = st.text_input("Location (e.g., -20°C box A1)", "")
            vol = st.text_input("Volume (e.g., 1 mL)", "")
        with c3:
            expiry = st.date_input("Expiry", value=dt.date.today()+dt.timedelta(days=180))
            submit = st.form_submit_button("Add")
    if submit and name:
        st.session_state["inventory"].append(dict(name=name, conc=conc, volume=vol, location=loc, expiry=str(expiry)))
        st.success("Added.")
    if st.session_state["inventory"]:
        st.dataframe(st.session_state["inventory"])

# ======================================================================
# 22) REAGENT STABILITY PREDICTOR
# ======================================================================
elif mode == "Reagent stability predictor":
    st.subheader("Reagent stability predictor")
    txt = st.text_area("Describe reagent & conditions", placeholder="e.g., Retinal in EtOH, room light, 4°C storage")
    if txt.strip():
        t = txt.lower()
        tips = []
        if "retinal" in t or "retinoic" in t:
            tips += ["Light-sensitive: wrap in foil/amber, use dry solvent, store −20°C."]
        if "dye" in t or "fluorescein" in t or "rhodamine" in t:
            tips += ["Avoid repeated freeze–thaw; consider aliquots."]
        if "enzyme" in t:
            tips += ["Keep on ice during set-up; glycerol may stabilize."]
        if "aqueous" in t and "dmso" in t:
            tips += ["Gradually pre-dilute in DMSO, then add to aqueous media to avoid precipitation."]
        if tips:
            for t in tips: st.write("•", t)
        else:
            st.info("No specific rule matched. Use general best practices.")

# ======================================================================
# 23) DILUTION SERIES VISUALIZER
# ======================================================================
elif mode == "Dilution series visualizer":
    st.subheader("Dilution series visualizer")
    start = st.number_input("Start concentration (µM)", value=100.0, min_value=0.0001)
    factor = st.number_input("Dilution factor", value=2.0, min_value=1.001)
    steps = st.number_input("Steps", value=8, min_value=1, step=1)
    concs = [start/(factor**i) for i in range(int(steps))]
    df = pd.DataFrame({"step": list(range(1,int(steps)+1)), "concentration_µM": concs})
    st.dataframe(df)
    st.line_chart(df.set_index("step"))

# ======================================================================
# 24) NOTEBOOK GENERATOR (PDF/MD)
# ======================================================================
elif mode == "Notebook generator (PDF/MD)":
    st.subheader("Notebook generator (PDF/MD)")
    title = st.text_input("Title", "Dilution of compound X")
    purpose = st.text_area("Purpose", "Prepare dilution series for IC50 assay.")
    steps = st.text_area("Steps (one per line)", "Label tubes\nPipette stock\nAdd buffer\nMix gently")
    content = f"# {title}\n\n## Purpose\n{purpose}\n\n## Steps\n" + "\n".join([f"- {s}" for s in steps.splitlines() if s.strip()])
    st.download_button("⬇ Download Markdown", data=content.encode("utf-8"), file_name="notebook_entry.md")
    if 'HAS_FPDF' in globals() and HAS_FPDF:
        if st.button("📄 Export as PDF"):
            pdf_bytes = make_pdf_report(title, [purpose, "", "Steps:"] + steps.splitlines())
            st.download_button("⬇ Download PDF", data=pdf_bytes, file_name="notebook_entry.pdf")
    else:
        st.info("Install `fpdf` to enable PDF export.")

# ======================================================================
# 25) OSMOLARITY CALCULATOR
# ======================================================================
elif mode == "Osmolarity calculator":
    st.subheader("Osmolarity calculator")
    st.caption("Enter solutes and concentrations (mM). i = van’t Hoff factor (auto for common solutes).")
    names_default = ["NaCl", "KCl", "Glucose"]
    with st.form("osm"):
        rows_n = st.number_input("Number of components", value=3, min_value=1, step=1)
        comps = []
        for i in range(rows_n):
            c1,c2 = st.columns([2,1])
            with c1:
                name = st.text_input(f"Name {i+1}", value=names_default[i] if i<len(names_default) else "")
            with c2:
                C = st.number_input(f"C (mM) {i+1}", value=100.0 if i==0 else 0.0, min_value=0.0)
            comps.append({"name": name, "C_mM": C})
        go = st.form_submit_button("Calculate")
    if go:
        Osm = osmolarity(comps)     # Osm/L
        st.success(f"Estimated osmolarity: **{Osm*1000:.1f} mOsm/L**")

# ======================================================================
# 26) SPECTROPHOTOMETRY TOOLBOX
# ======================================================================
elif mode == "Spectrophotometry toolbox":
    st.subheader("Spectrophotometry toolbox (standard curve)")
    st.caption("Upload CSV with two columns: concentration, absorbance")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
        st.dataframe(df.head())
        if {"concentration","absorbance"}.issubset({c.lower() for c in df.columns}):
            # robust column access
            colmap = {c.lower(): c for c in df.columns}
            x = df[colmap["concentration"]].values.tolist()
            y = df[colmap["absorbance"]].values.tolist()
            fit = simple_linreg(x,y)
            if fit:
                st.write(f"Slope: **{fit['slope']:.6f}**, Intercept: **{fit['intercept']:.6f}**, R²: **{fit['r2']:.4f}**")
                st.info("Use: concentration = (A - intercept) / slope")
            else:
                st.warning("Could not fit line — check data.")
        else:
            st.error("CSV must have columns named 'concentration' and 'absorbance'.")

# ======================================================================
# 27) SOLUTION DENSITY CONVERTER
# ======================================================================
elif mode == "Solution density converter":
    st.subheader("Solution density converter")
    sol = st.selectbox("Solvent", ["water","ethanol_100%","ethanol_70%","glycerol_100%","dmso_100%","acetone_100%","custom"])
    if sol=="custom":
        dens = st.number_input("Density (g/mL)", value=1.0, min_value=0.5, max_value=2.0)
    else:
        dens = DENSITY.get(sol, 1.0)
    st.write(f"Using density: **{dens:.3f} g/mL**")
    direction = st.radio("Convert", ["% w/v → g/L and M", "M → % w/v (approx)"])
    if direction == "% w/v → g/L and M":
        percent = st.number_input("% w/v (g/100 mL)", value=10.0, min_value=0.0)
        mw = st.number_input("MW (g/mol)", value=180.16, min_value=1.0)
        g_per_L = percent * 10.0  # g/100 mL → g/L
        M = g_per_L / mw
        st.success(f"{percent}% w/v ≈ **{g_per_L:.1f} g/L** (≈ **{M:.3f} M**)")
    else:
        M = st.number_input("Molarity (M)", value=1.0, min_value=0.0)
        mw = st.number_input("MW (g/mol)", value=180.16, min_value=1.0)
        g_per_L = M * mw
        percent = g_per_L / 10.0
        st.success(f"{M} M ≈ **{g_per_L:.1f} g/L** (≈ **{percent:.1f}% w/v**)")

# ======================================================================
# 28) REAGENT COMPATIBILITY CHECKER
# ======================================================================
elif mode == "Reagent compatibility checker":
    st.subheader("Reagent compatibility checker")
    txt = st.text_area("Describe your planned mix", placeholder="e.g., Phosphate buffer with calcium chloride and ethanol…")
    if st.button("Check"):
        hits = check_compatibility(txt)
        if hits:
            for h in hits:
                st.warning(h)
        else:
            st.success("No issues found in our simple rule set. (Always double-check your specific system.)")

# ======================================================================
# 29) STORAGE / STABILITY
# ======================================================================
else:  # "Storage / stability helper"
    st.subheader("Storage / stability helper")

    name = st.text_input("Compound / solution name", "")
    storage_dict = {
        "retinal": "Protect from light, dissolve in dry EtOH or DMSO, aliquot, store at -20°C or below.",
        "retinoic": "Light-sensitive, store at -20°C, use fresh aliquots.",
        "ampicillin": "Store stock at -20°C, avoid repeated freeze–thaw.",
        "pbs": "Room temp or 4°C, 1 month.",
        "tris": "Room temp, 1 month.",
        "pfa": "4°C, protected from light, check for precipitate.",
    }

    out = None
    for key, val in storage_dict.items():
        if key in name.lower():
            out = val
            break

    if out:
        st.success(out)
    else:
        st.info(
            "No specific rule found. General rule: store at 4°C for short term, -20°C for long term, "
            "protect from light if colored/retinoid."
        )

# ------------------------------------------------------------
# Tier-5: function registry for chat & batch
# ------------------------------------------------------------
def calc_single(stock_conc, target_conc, final_ul, vehicle_frac):
    v1_ul = (target_conc * final_ul) / stock_conc
    solvent_ul = max(final_ul - v1_ul, 0.0)
    vehicle_percent = (v1_ul * vehicle_frac / final_ul) * 100 if final_ul > 0 else 0
    return {
        "add_stock_ul": round(v1_ul, 4),
        "add_solvent_ul": round(solvent_ul, 4),
        "vehicle_percent": round(vehicle_percent, 6),
    }


def calc_unit_mgml_to_mM(mg_per_ml, mw):
    return {"mM": (mg_per_ml * 1000.0) / mw}


def calc_unit_mM_to_mgml(mM, mw):
    return {"mg_per_ml": (mM * mw) / 1000.0}


CALC_REGISTRY: Dict[str, Callable[..., Dict[str, Any]]] = {
    "single_dilution": calc_single,
    "mgml_to_mM": calc_unit_mgml_to_mM,
    "mM_to_mgml": calc_unit_mM_to_mgml,
}

# ------------------------------------------------------------
# Tier-5 UI: Chat • Batch • Image • Cloud • API keys
# ------------------------------------------------------------
st.markdown("---")
st.header("⚡ Tier-5: Chat • Batch • Image • Cloud • API")

if plan != TIER5_REQUIRED_PLAN:
    st.info(f"Tier-5 features require **{TIER5_REQUIRED_PLAN}**. Contact admin to upgrade.")
else:
    tab_chat, tab_batch, tab_image, tab_cloud, tab_keys = st.tabs(
        ["Chat interface", "Batch calculator", "Image input", "DataLens Cloud", "API keys"]
    )

    # --- (A) Chat interface ---
    with tab_chat:
        st.subheader("Chat with your Lab Assistant")

        if "chat_session_id" not in st.session_state:
            title = f"Session {datetime.utcnow().isoformat(timespec='seconds')}"
            try:
                res = (
                    supabase.table("chat_sessions")
                    .insert({"user_id": user.id, "title": title})
                    .execute()
                )
                st.session_state["chat_session_id"] = res.data[0]["id"]
            except Exception:
                st.session_state["chat_session_id"] = None

        chat_session_id = st.session_state.get("chat_session_id")

        # load history
        history = []
        if chat_session_id:
            try:
                msgs = (
                    supabase.table("chat_messages")
                    .select("*")
                    .eq("session_id", chat_session_id)
                    .order("created_at", desc=False)
                    .execute()
                )
                history = msgs.data or []
            except Exception:
                history = []

        for m in history:
            with st.chat_message(m["sender"]):
                st.markdown(m["content"])
                if m.get("tool_output"):
                    st.caption(f"Tool: {m.get('tool_called')}")
                    st.json(m["tool_output"])

        prompt = st.chat_input(
            "e.g., Make 10 µM from 10 mM in 300 µL; or Convert 1 mg/mL to mM (MW 284.44)"
        )
        if prompt:
            with st.chat_message("user"):
                st.markdown(prompt)
            save_chat_msg(chat_session_id, "user", prompt)

            tool, args = None, {}
            t = prompt.lower()

            try:
                # mg/mL → mM
                if "mg/ml" in t and "mm" in t and "convert" in t:
                    vals = [float(x) for x in t.replace("/", " ").split() if x.replace(".", "", 1).isdigit()]
                    if len(vals) >= 2:
                        tool = "mgml_to_mM"
                        args = {"mg_per_ml": vals[0], "mw": vals[1]}

                # single dilution
                elif "make" in t and "from" in t and ("µm" in t or "um" in t) and ("mm" in t):
                    nums = [float(x) for x in t.replace("µ", "u").split() if x.replace(".", "", 1).isdigit()]
                    if len(nums) >= 2:
                        target, stock = nums[0], nums[1]
                        vol = nums[2] if len(nums) >= 3 else well_volume
                        tool = "single_dilution"
                        args = {
                            "stock_conc": stock,
                            "target_conc": target,
                            "final_ul": vol,
                            "vehicle_frac": vehicle_frac,
                        }

                if tool is None:
                    msg = (
                        "I can run:\n"
                        "- **single_dilution** – ‘Make 10 µM from 10 mM in 300 µL’\n"
                        "- **mg/mL↔mM** – ‘Convert 1 mg/mL to mM (MW 284.44)’"
                    )
                    with st.chat_message("assistant"):
                        st.markdown(msg)
                    save_chat_msg(chat_session_id, "assistant", msg)
                else:
                    out = CALC_REGISTRY[tool](**args)
                    with st.chat_message("assistant"):
                        st.markdown(f"**Tool:** `{tool}`")
                        st.json(out)
                    save_chat_msg(
                        chat_session_id,
                        "assistant",
                        f"Ran `{tool}`",
                        tool_called=tool,
                        tool_input=args,
                        tool_output=out,
                    )
                    save_run_to_cloud(tool, args, out, user_id=user.id)
            except Exception as e:
                err = f"Error: {e}"
                with st.chat_message("assistant"):
                    st.error(err)
                save_chat_msg(chat_session_id, "assistant", err)

    # --- (B) Batch calculator ---
    with tab_batch:
        st.subheader("Batch calculator (CSV)")
        st.code(
            """mode,stock_mM,target_mM,final_ul
               single_dilution,10,0.01,300

               mode,mg_per_ml,mw
               mgml_to_mM,1,284.44

               mode,mM,mw
               mM_to_mgml,10,284.44
            """,
            language="csv",
        )

        up = st.file_uploader("Upload CSV", type=["csv"], key="batch_csv")
        if up is not None:
            df_in = pd.read_csv(up)
            out_rows = []
            for _, row in df_in.iterrows():
                try:
                    m = row["mode"]
                    if m == "single_dilution":
                        out = CALC_REGISTRY["single_dilution"](
                            stock_conc=float(row["stock_mM"]),
                            target_conc=float(row["target_mM"]),
                            final_ul=float(row["final_ul"]),
                            vehicle_frac=vehicle_frac,
                        )
                    elif m == "mgml_to_mM":
                        out = CALC_REGISTRY["mgml_to_mM"](
                            mg_per_ml=float(row["mg_per_ml"]),
                            mw=float(row["mw"]),
                        )
                    elif m == "mM_to_mgml":
                        out = CALC_REGISTRY["mM_to_mgml"](mM=float(row["mM"]), mw=float(row["mw"]))
                    else:
                        out = {"error": f"Unsupported mode: {m}"}
                    out_rows.append({**row.to_dict(), **out})
                except Exception as e:
                    out_rows.append({**row.to_dict(), "error": str(e)})

            df_out = pd.DataFrame(out_rows)
            st.dataframe(df_out)
            st.download_button(
                "⬇ Download results",
                df_out.to_csv(index=False).encode("utf-8"),
                "batch_results.csv",
                "text/csv",
            )
            save_run_to_cloud("batch", {"rows": len(df_in)}, {"rows": len(df_out)}, user_id=user.id)

    # --- (C) Image input / OCR ---
    with tab_image:
        st.subheader("Image input (labels/plates)")

        if not HAS_OCR:
            st.info(
                "OCR is not available on this deployment (Tesseract engine missing). "
                "You can still upload images under **DataLens Cloud** to store them, "
                "and run OCR locally on your own computer."
            )

        image_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"], key="image_ocr")

        if image_file is not None:
            if HAS_OCR:
                img = Image.open(image_file)
                st.image(img, caption="Uploaded")
                try:
                    text = pytesseract.image_to_string(img)
                    st.text_area("OCR text", value=text, height=200)
                    st.caption("Tip: copy values into the calculators above.")
                except Exception as e:
                    st.error(f"OCR failed: {e}")
            else:
                st.warning(
                    "Image uploaded, but OCR is disabled here because Tesseract is not installed "
                    "on the server. Use a local copy of the app for OCR."
                )

    # --- (D) DataLens Cloud ---
    with tab_cloud:
       st.subheader("Cloud uploads & recent runs")

    # 1) Upload to storage + log in uploads table
    cfile = st.file_uploader(
        "Upload to cloud",
        key="cloud_up",
        type=["png", "jpg", "jpeg", "csv", "pdf"],
    )
    if cfile:
        content = cfile.read()
        path = f"{int(time.time())}_{cfile.name}"
        try:
            # 👇 use the same bucket name you actually created
            supabase.storage.from_("datalens-secure-uploads").upload(path, content)
            supabase.table("uploads").insert(
                {
                    "user_id": user.id,
                    "filename": cfile.name,
                    "mime_type": cfile.type,
                    "storage_path": path,
                    "bytes": len(content),
                }
            ).execute()
            st.success(f"Uploaded: {cfile.name}")
        except Exception as e:
            st.error(f"Upload failed: {e}")

    # 2) List recent uploads for this user (with signed URLs)
    try:
        up_res = (
            supabase.table("uploads")
            .select("id, filename, mime_type, storage_path, created_at")
            .eq("user_id", user.id)
            .order("created_at", desc=True)
            .limit(20)
            .execute()
        )
        uploads = up_res.data or []

        st.markdown("### Your recent uploads")

        if not uploads:
            st.info("No uploads yet.")
        else:
            for row in uploads:
                # 👇 bucket name must match the one you upload to
                res = supabase.storage.from_("datalens-secure-uploads").create_signed_url(
                    row["storage_path"], 3600  # 1 hour
                )
                url = res["signedURL"]  # or res["signed_url"] depending on client version

                if row["mime_type"].startswith("image/"):
                    st.image(url, caption=row["filename"], width=300)
                else:
                    st.markdown(f"📄 [{row['filename']}]({url})")

    except Exception as e:
        st.error(f"Could not load uploads: {e}")

    # 3) Show recent calculation runs
    try:
        runs = (
            supabase.table("runs")
            .select("*")
            .eq("user_id", user.id)
            .order("created_at", desc=True)
            .limit(20)
            .execute()
        )
        st.markdown("### Recent calculator runs")
        st.dataframe(pd.DataFrame(runs.data) if runs.data else pd.DataFrame())
    except Exception:
        st.info("No runs yet.")

    # --- (E) API keys ---
    with tab_keys:
       st.subheader("API keys (for external access)")
    new_name = st.text_input("Key name", "")

    if st.button("Generate API key") and new_name:
        raw = "dlk_" + hashlib.sha256(f"{time.time()}_{new_name}".encode()).hexdigest()[:40]
        try:
            supabase.table("api_keys").insert(
                {
                    "user_id": user.id,          # 👈 important
                    "name": new_name,
                    "key_hash": hash_api_key(raw),
                }
            ).execute()
            st.success(f"**Save this key now**: `{raw}`")
            st.caption("Only the hash is stored; you won’t see the raw key again.")
        except Exception as e:
            st.error(f"Could not create key: {e}")

    try:
        keys = (
            supabase.table("api_keys")
            .select("id,name,created_at,last_used_at")
            .eq("user_id", user.id)           # 👈 only your keys
            .execute()
        )
        st.write("Existing keys")
        st.dataframe(pd.DataFrame(keys.data) if keys.data else pd.DataFrame())
    except Exception:
        st.info("No keys yet.")

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background: #0f172a;
        color: white;
        text-align: center;
        padding: 6px 0;
        font-size: 0.8rem;
        z-index: 9999;
    }
    </style>
    <div class="footer">
        © 2025 DataLens.Tools
    </div>
    """,
    unsafe_allow_html=True,
)
