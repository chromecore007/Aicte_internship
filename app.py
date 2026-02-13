import streamlit as st
import pickle
import numpy as np

# Load model & vectorizer
model = pickle.load(open("mental_health_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Mental Health AI",
    page_icon="🧠",
    layout="centered"
)

st.markdown("""
<style>
/* Hide Streamlit default menu (Deploy, Settings, etc.) */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.main {
    max-width: 720px;
    margin: auto;
}
textarea {
    font-size: 16px !important;
}
.result-box {
    padding: 18px;
    border-radius: 12px;
    margin-top: 20px;
}
.low {
    background-color: #1f6f3e;
}
.medium {
    background-color: #8a6d1f;
}
.high {
    background-color: #7a1f1f;
}
.footer {
    margin-top: 40px;
    font-size: 13px;
    opacity: 0.7;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("## 🧠 Mental Health Risk Detection")
st.markdown(
    "Early identification of mental health risk using AI.  \n"
    "<small>⚠️ Not a medical diagnosis</small>",
    unsafe_allow_html=True
)

# ---------- INPUT ----------
user_text = st.text_area(
    "How are you feeling today?",
    height=140,
    placeholder="Example: I feel stressed and exhausted lately..."
)

# ---------- BUTTON ----------
analyze = st.button("Analyze")

# ---------- PREDICTION ----------
if analyze:
    if user_text.strip() == "":
        st.warning("Please write something to analyze.")
    else:
        text_vector = vectorizer.transform([user_text])
        prediction = model.predict(text_vector)[0]
        prob = model.predict_proba(text_vector)[0]
        confidence = round(np.max(prob) * 100, 2)

        # HIGH / MEDIUM RISK
        if prediction == 1:
            if confidence >= 75:
                st.markdown(
                    f"""
                    <div class="result-box high">
                        🔴 <b>High Risk Detected</b><br>
                        Confidence: {confidence}%
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-box medium">
                        🟡 <b>Moderate Risk Detected</b><br>
                        Confidence: {confidence}%
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("### Support Suggestions")
            st.markdown("""
- Talk to someone you trust  
- Avoid isolation  
- Maintain a healthy sleep routine  
- Consider consulting a mental health professional  

📞 **India Mental Health Helpline:** 9152987821
""")

        # LOW RISK
        else:
            st.markdown(
                f"""
                <div class="result-box low">
                    🟢 <b>Low Risk Detected</b><br>
                    Confidence: {confidence}%
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("### Well-being Tips")
            st.markdown("""
- Maintain healthy daily habits  
- Stay socially connected  
- Continue activities you enjoy  
""")

# ---------- FOOTER ----------
st.markdown(
    """
    <div class="footer">
        Built with ❤️ using Machine Learning & NLP  
    </div>
    """,
    unsafe_allow_html=True
)
