# ui_streamlit.py
import streamlit as st
import requests
from PIL import Image
import io
import os

BACKEND_URL = "http://127.0.0.1:8000/process"

st.set_page_config(page_title="AI EHR Imaging", layout="centered")
st.title("🩺 AI-Enhanced EHR Imaging")

uploaded = st.file_uploader("Upload X-ray image (jpg/png)", type=["jpg","jpeg","png"])
age = st.text_input("Age", "45")
gender = st.selectbox("Gender", ["Male","Female","Other"])
symptom = st.text_input("Symptoms", "Cough, fever")
diagnosis = st.text_input("Diagnosis", "Pneumonia")
lab = st.text_input("Lab results", "WBC high")

if uploaded:
    st.image(uploaded, caption="Original X-ray", use_container_width=True)

if st.button("Process (send to backend)"):
    if not uploaded:
        st.error("Upload an image first")
    else:
        files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
        data = {
            "age": age,
            "gender": gender,
            "symptom": symptom,
            "diagnosis": diagnosis,
            "lab": lab
        }
        try:
            with st.spinner("Processing..."):
                r = requests.post(BACKEND_URL, files=files, data=data, timeout=60)
                r.raise_for_status()
                res = r.json()
            st.success("Processed ✅")
            st.write("**ICD-10 suggestion:**", res.get("icd10"))
            st.write("**Clinical summary:**")
            st.write(res.get("clinical_summary"))
            # show enhanced image
            enhanced_path = res.get("enhanced_image_path")
            if enhanced_path and os.path.exists(enhanced_path):
                st.image(enhanced_path, caption="Enhanced X-ray", use_container_width=True)
            else:
                st.warning("Enhanced image returned but file not found on UI machine.")
        except Exception as e:
            st.error(f"Request failed: {e}")
