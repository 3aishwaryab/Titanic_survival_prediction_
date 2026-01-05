"""Streamlit frontend for Titanic Survival Prediction

Usage:
  pip install streamlit
  streamlit run frontend/streamlit_app.py
"""
import streamlit as st
import requests
import json

st.set_page_config(page_title="Titanic Survival Demo", layout="centered")
st.title("Titanic Survival Prediction")

st.markdown("Fill in a passenger profile and press **Predict**. The app will call the backend `/predict` endpoint and show the model's result.")

with st.sidebar.form(key='api_settings'):
    st.header("API settings")
    host = st.text_input("Host", value="127.0.0.1")
    port = st.number_input("Port", value=8000, min_value=1, max_value=65535)
    submit_api = st.form_submit_button("Save")

base_url = f"http://{host}:{port}"

with st.form(key='passenger_form'):
    st.subheader("Passenger details")
    pclass = st.selectbox("Pclass", [1, 2, 3], index=0)
    sex = st.selectbox("Sex", ["female", "male"], index=0)
    age = st.number_input("Age", min_value=0.0, value=30.0, format="%.1f")
    sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", min_value=0, value=0)
    parch = st.number_input("Parents/Children Aboard (Parch)", min_value=0, value=0)
    fare = st.number_input("Fare", min_value=0.0, value=32.0, format="%.2f")
    embarked = st.selectbox("Embarked", ["S", "C", "Q"], index=0)
    name = st.text_input("Name", value="Doe, Ms. Jane")

    do_predict = st.form_submit_button("Get Prediction")

if do_predict:
    payload = {
        "Pclass": int(pclass),
        "Sex": sex,
        "Age": float(age),
        "SibSp": int(sibsp),
        "Parch": int(parch),
        "Fare": float(fare),
        "Embarked": embarked,
        "Name": name
    }

    url = f"{base_url}/predict"
    st.write(f"Calling {url} with payload:")
    st.json(payload)

    with st.spinner("Contacting API and waiting for prediction..."):
        try:
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    pred = data.get('prediction')
                    if pred is None:
                        st.error(f"Unexpected response: {data}")
                    else:
                        st.success(f"Prediction: {'Survived' if int(pred) == 1 else 'Did not survive'} ({int(pred)})")
                        st.write("Full response:")
                        st.json(data)
                except Exception as e:
                    st.error(f"Failed to parse JSON response: {e}")
            elif resp.status_code == 405:
                st.warning("Method Not Allowed. Ensure you're sending a POST with a JSON payload to /predict.")
            else:
                st.error(f"Server error ({resp.status_code}): {resp.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")
