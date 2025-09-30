import streamlit as st
import requests
from PIL import Image

url = "https://binnyman-simp.hf.space/classify"

st.title("Simpsons")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify"):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        resp = requests.post(url, files=files)
        if resp.status_code == 200:
            result = resp.json()
            st.success(f"predicted_label:{result['predicted_label']}")
        else:
            st.error("Error")
