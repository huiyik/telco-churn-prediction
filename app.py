import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 1. LOAD MODEL
model = joblib.load('churn_model.pkl')
model_columns = joblib.load('model_columns.pkl')

# 2. PAGE CONFIG
st.set_page_config(page_title="Telco Churn Prediction System", page_icon="📡", layout="wide")

# 3. CUSTOM CSS
st.markdown("""
    <style>
        .stApp { background-color: #0f1623; }
        .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }

        .main-title {
            text-align: center;
            font-size: 2.4rem;
            font-weight: 800;
            color: #ffffff;
            letter-spacing: 1px;
            margin-bottom: 4px;
        }
        .main-subtitle {
            text-align: center;
            font-size: 0.95rem;
            color: #8a9bbf;
            margin-bottom: 0;
        }
        .title-badge {
            display: inline-block;
            background: #1e3a5f;
            color: #60a5fa;
            border-radius: 20px;
            padding: 4px 14px;
            font-size: 0.8rem;
            font-weight: 600;
            margin: 4px;
        }
        .card-title {
            font-size: 1rem;
            font-weight: 700;
            color: #60a5fa;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 16px;
        }
        .stSelectbox label, .stNumberInput label {
            color: #94a3b8 !important;
            font-size: 0.85rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.5px !important;
        }
        .stSelectbox > div > div, .stNumberInput > div > div > input {
            background-color: #1e2d47 !important;
            border: 1px solid #2d4a6e !important;
            border-radius: 8px !important;
            color: #ffffff !important;
        }
        .stButton > button {
            background: linear-gradient(135deg, #3b82f6, #1d4ed8);
            color: #ffffff;
            font-weight: 700;
            font-size: 1.05rem;
            border-radius: 12px;
            height: 3.2rem;
            border: none;
            letter-spacing: 0.5px;
            box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #60a5fa, #3b82f6);
            box-shadow: 0 6px 20px rgba(59, 130, 246, 0.5);
        }
        .result-high {
            background: linear-gradient(135deg, #2d0a0a, #450a0a);
            border: 2px solid #ef4444;
            border-radius: 14px;
            padding: 20px 24px;
            text-align: center;
        }
        .result-moderate {
            background: linear-gradient(135deg, #1c1208, #2d1f08);
            border: 2px solid #f59e0b;
            border-radius: 14px;
            padding: 20px 24px;
            text-align: center;
        }
        .result-safe {
            background: linear-gradient(135deg, #071a0f, #0d2a18);
            border: 2px solid #22c55e;
            border-radius: 14px;
            padding: 20px 24px;
            text-align: center;
        }
        .result-label-high { color: #ef4444; font-size: 1.6rem; font-weight: 800; }
        .result-label-moderate { color: #f59e0b; font-size: 1.6rem; font-weight: 800; }
        .result-label-safe { color: #22c55e; font-size: 1.6rem; font-weight: 800; }
        .result-desc { color: #94a3b8; font-size: 0.9rem; margin-top: 6px; }
        .prob-card {
            background: #172035;
            border-radius: 14px;
            padding: 20px 24px;
            text-align: center;
            border: 1px solid #1e2d47;
        }
        .prob-number { font-size: 3.5rem; font-weight: 800; color: #ffffff; line-height: 1; }
        .prob-label { color: #60a5fa; font-size: 0.85rem; font-weight: 600; margin-top: 4px; letter-spacing: 1px; text-transform: uppercase; }
        .factor-section-red {
            background: #1f0d0d;
            border: 1px solid #ef444444;
            border-left: 4px solid #ef4444;
            border-radius: 10px;
            padding: 16px 18px;
            margin-bottom: 12px;
        }
        .factor-section-green {
            background: #0d1f12;
            border: 1px solid #22c55e44;
            border-left: 4px solid #22c55e;
            border-radius: 10px;
            padding: 16px 18px;
            margin-bottom: 12px;
        }
        .factor-header-red {
            color: #ef4444;
            font-weight: 700;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }
        .factor-header-green {
            color: #22c55e;
            font-weight: 700;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }
        .factor-chip-red {
            display: inline-block;
            background: #ef444422;
            border: 1px solid #ef444466;
            color: #fca5a5;
            border-radius: 20px;
            padding: 4px 12px;
            font-size: 0.85rem;
            margin: 3px;
        }
        .factor-chip-green {
            display: inline-block;
            background: #22c55e22;
            border: 1px solid #22c55e66;
            color: #86efac;
            border-radius: 20px;
            padding: 4px 12px;
            font-size: 0.85rem;
            margin: 3px;
        }
        .action-card {
            background: #0f1e35;
            border: 1px solid #3b82f644;
            border-left: 4px solid #3b82f6;
            border-radius: 10px;
            padding: 16px 18px;
            margin-top: 12px;
        }
        .action-label {
            color: #60a5fa;
            font-weight: 700;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 6px;
        }
        .action-text { color: #cbd5e1; font-size: 0.95rem; line-height: 1.5; }
        .info-banner {
            background: #1e2d47;
            border-radius: 8px;
            padding: 10px 16px;
            color: #60a5fa;
            font-size: 0.85rem;
            margin-top: 4px;
        }
        hr { border-color: #1e2d47 !important; margin: 1.5rem 0 !important; }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ── HELPER ──────────────────────────────────────────────
def clean_name(name):
    replacements = {
        'InternetService_Fiber optic': 'Internet Service: Fiber Optic',
        'InternetService_No': 'Internet Service: None',
        'OnlineSecurity_No internet service': 'Online Security: No Internet',
        'OnlineSecurity_Yes': 'Online Security: Active',
        'TechSupport_No internet service': 'Tech Support: No Internet',
        'TechSupport_Yes': 'Tech Support: Active',
        'Contract_One year': 'Contract: One Year',
        'Contract_Two year': 'Contract: Two Year',
        'PaperlessBilling_Yes': 'Paperless Billing: Yes',
        'PaymentMethod_Credit card (automatic)': 'Payment: Credit Card',
        'PaymentMethod_Electronic check': 'Payment: Electronic Check',
        'PaymentMethod_Mailed check': 'Payment: Mailed Check',
        'MonthlyCharges': 'Monthly Charges',
        'tenure': 'Tenure (Months)',
    }
    return replacements.get(name, name.replace('_', ' ').strip())

# ── TITLE ────────────────────────────────────────────────
st.markdown("<div class='main-title'>📡 Telco Churn Prediction System</div>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align:center; margin-top:8px; margin-bottom:4px;'>
        <span class='title-badge'>🤖 Logistic Regression</span>
        <span class='title-badge'>✅ 80% Accuracy</span>
        <span class='title-badge'>👥 7,043 Customers</span>
    </div>
    <div class='main-subtitle'>Enter customer details below to assess churn risk in real time</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

# ── INPUT SECTION ────────────────────────────────────────
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("<div class='card-title'>🧾 Account Details</div>", unsafe_allow_html=True)
    tenure = st.number_input("How long has this customer been with us? (months)", min_value=0, max_value=100, value=5)
    contract_type = st.selectbox("What type of contract does the customer have?", ["Month-to-month", "One year", "Two year"])
    monthly_charges = st.number_input("How much does the customer pay monthly? (RM)", min_value=0.0, value=50.0)
    payment_method = st.selectbox("How does the customer pay?", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

with col2:
    st.markdown("<div class='card-title'>📶 Service Details</div>", unsafe_allow_html=True)
    internet_service = st.selectbox("What internet service does the customer use?", ["DSL", "Fiber optic", "No"])

    # Smart logic — if no internet, auto-set security and tech support
    if internet_service == "No":
        st.markdown("<div class='info-banner'>ℹ️ Online Security and Tech Support are not applicable — this customer has no internet service.</div>", unsafe_allow_html=True)
        online_security = "No internet service"
        tech_support = "No internet service"
    else:
        online_security = st.selectbox("Does the customer have online security?", ["No", "Yes"])
        tech_support = st.selectbox("Does the customer have tech support?", ["No", "Yes"])

    paperless_billing = st.selectbox("Is the customer on paperless billing?", ["Yes", "No"])

st.markdown("---")
predict_clicked = st.button("🔍 Analyse Customer Churn Risk", type="primary", use_container_width=True)

# ── PREDICTION ────────────────────────────────────────────
if predict_clicked:

    input_data = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'InternetService_Fiber optic': 1 if internet_service == "Fiber optic" else 0,
        'InternetService_No': 1 if internet_service == "No" else 0,
        'OnlineSecurity_No internet service': 1 if online_security == "No internet service" else 0,
        'OnlineSecurity_Yes': 1 if online_security == "Yes" else 0,
        'TechSupport_No internet service': 1 if tech_support == "No internet service" else 0,
        'TechSupport_Yes': 1 if tech_support == "Yes" else 0,
        'Contract_One year': 1 if contract_type == "One year" else 0,
        'Contract_Two year': 1 if contract_type == "Two year" else 0,
        'PaperlessBilling_Yes': 1 if paperless_billing == "Yes" else 0,
        'PaymentMethod_Credit card (automatic)': 1 if payment_method == "Credit card (automatic)" else 0,
        'PaymentMethod_Electronic check': 1 if payment_method == "Electronic check" else 0,
        'PaymentMethod_Mailed check': 1 if payment_method == "Mailed check" else 0,
    }

    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    churn_percent = round(probability * 100, 1)

    # Contributions
    coefficients = model.coef_[0]
    contributions = coefficients * input_df.values[0]
    contrib_df = pd.DataFrame({'Feature': model_columns, 'Contribution': contributions})
    contrib_df['Abs'] = contrib_df['Contribution'].abs()
    contrib_df = contrib_df[contrib_df['Abs'] > 0].sort_values('Abs', ascending=False).head(10)

    risk_factors = contrib_df[contrib_df['Contribution'] > 0]['Feature'].tolist()
    protect_factors = contrib_df[contrib_df['Contribution'] < 0]['Feature'].tolist()

    # ── RESULTS ───────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>📊 Analysis Result</div>", unsafe_allow_html=True)

    r1, r2, r3 = st.columns([1.2, 1, 1.8], gap="large")

    with r1:
        prob_color = "#ef4444" if churn_percent >= 70 else "#f59e0b" if prediction == 1 else "#22c55e"
        st.markdown(f"""
            <div class='prob-card'>
                <div class='prob-label'>Churn Probability</div>
                <div class='prob-number' style='color:{prob_color};'>{churn_percent}%</div>
                <div style='color:#475569; font-size:0.8rem; margin-top:8px;'>Based on 7,043 customer records</div>
            </div>
        """, unsafe_allow_html=True)

    with r2:
        if prediction == 1 and churn_percent >= 70:
            st.markdown("""
                <div class='result-high'>
                    <div style='font-size:2.2rem;'>🚨</div>
                    <div class='result-label-high'>HIGH RISK</div>
                    <div class='result-desc'>This customer is very likely to leave.</div>
                </div>
            """, unsafe_allow_html=True)
        elif prediction == 1:
            st.markdown("""
                <div class='result-moderate'>
                    <div style='font-size:2.2rem;'>⚠️</div>
                    <div class='result-label-moderate'>MODERATE RISK</div>
                    <div class='result-desc'>This customer shows signs of churning.</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class='result-safe'>
                    <div style='font-size:2.2rem;'>✅</div>
                    <div class='result-label-safe'>LOW RISK</div>
                    <div class='result-desc'>This customer is likely to stay.</div>
                </div>
            """, unsafe_allow_html=True)

    with r3:
        if prediction == 1 and churn_percent >= 70:
            action = "This customer is at serious risk of leaving. We strongly recommend reaching out immediately with a personalised retention offer — consider a discounted long-term contract or an exclusive loyalty bundle."
        elif prediction == 1:
            action = "This customer shows moderate signs of churn. Consider enrolling them in a loyalty rewards programme or proactively checking in on their service satisfaction."
        else:
            action = "This customer appears stable. Continue delivering quality service. Periodic check-ins and reward programmes can further strengthen their loyalty."

        st.markdown(f"""
            <div class='action-card'>
                <div class='action-label'>💡 Recommended Action</div>
                <div class='action-text'>{action}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.progress(int(churn_percent), text=f"Risk Level: {churn_percent}%")

    # ── FACTORS ───────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>🔍 What's Driving This Prediction?</div>", unsafe_allow_html=True)

    f1, f2 = st.columns(2, gap="large")

    with f1:
        chips_red = "".join([f"<span class='factor-chip-red'>{clean_name(f)}</span>" for f in risk_factors]) if risk_factors else "<span style='color:#475569;'>None detected</span>"
        st.markdown(f"""
            <div class='factor-section-red'>
                <div class='factor-header-red'>⬆ Pushing Towards Churn</div>
                <div>{chips_red}</div>
            </div>
        """, unsafe_allow_html=True)

    with f2:
        chips_green = "".join([f"<span class='factor-chip-green'>{clean_name(f)}</span>" for f in protect_factors]) if protect_factors else "<span style='color:#475569;'>None detected</span>"
        st.markdown(f"""
            <div class='factor-section-green'>
                <div class='factor-header-green'>⬇ Holding the Customer Back</div>
                <div>{chips_green}</div>
            </div>
        """, unsafe_allow_html=True)

    # ── CHART ─────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>📈 Factor Contribution Chart</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#475569; font-size:0.85rem; margin-top:-10px;'>Shows how much each factor pushes the prediction towards or away from churn</p>", unsafe_allow_html=True)

    chart_df = contrib_df.copy()
    chart_df['Feature'] = chart_df['Feature'].apply(clean_name)
    chart_df = chart_df.sort_values('Contribution', ascending=True)
    colors = ['#ef4444' if x > 0 else '#22c55e' for x in chart_df['Contribution']]

    fig, ax = plt.subplots(figsize=(12, max(4, len(chart_df) * 0.7)))
    fig.patch.set_facecolor('#172035')
    ax.set_facecolor('#172035')

    bars = ax.barh(chart_df['Feature'], chart_df['Contribution'],
                   color=colors, height=0.5, edgecolor='none')

    for bar, val in zip(bars, chart_df['Contribution']):
        xpos = bar.get_width() + (0.01 if val > 0 else -0.01)
        ha = 'left' if val > 0 else 'right'
        ax.text(xpos, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', ha=ha,
                color='#94a3b8', fontsize=8)

    ax.set_xlabel('Contribution Score', color='#8a9bbf', fontsize=10)
    ax.set_title('Factor Contribution to Churn Prediction', color='#ffffff',
                 fontsize=13, fontweight='bold', pad=15)
    ax.tick_params(colors='#94a3b8', labelsize=10)
    ax.spines['bottom'].set_color('#1e2d47')
    ax.spines['left'].set_color('#1e2d47')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axvline(x=0, color='#2d4a6e', linewidth=1.2, linestyle='--')
    ax.grid(axis='x', color='#1e2d47', linewidth=0.5, alpha=0.5)

    red_patch = mpatches.Patch(color='#ef4444', label='Increases Churn Risk')
    green_patch = mpatches.Patch(color='#22c55e', label='Decreases Churn Risk')
    ax.legend(handles=[red_patch, green_patch], facecolor='#172035',
              edgecolor='#1e2d47', labelcolor='#94a3b8', fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)

    # ── FOOTER ────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align:center; color:#334155; font-size:0.8rem; padding:16px 0;'>
            Telco Churn Prediction System &nbsp;|&nbsp; Tang Hui Yi 299390 &nbsp;|&nbsp; SQQPK 3123 &nbsp;|&nbsp; UUM
        </div>
    """, unsafe_allow_html=True)
