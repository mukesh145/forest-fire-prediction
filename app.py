import streamlit as st
import requests
import pickle
import numpy as np
import os
from dotenv import load_dotenv
import datetime

load_dotenv()
now = datetime.datetime.now()


with open("model.pkl", "rb") as file:
    model = pickle.load(file)


def fetch_weather_data():
    lat = os.getenv('lat') 
    lon = os.getenv('lon')

    url = os.getenv('url')
    ACCESS_KEY = os.getenv('ACCESS_KEY')

    params = {
        'access_key': ACCESS_KEY,
        'query': f'{lat},{lon}',
        'units': 'm'  
    }

    response = requests.get(url, params=params)
    data = response.json()

    temperature = data['current']['temperature']
    humidity = data['current']['humidity']
    wind_speed = data['current']['wind_speed']
    precipitation = data['current']['precip']

    details = [now.month-1,temperature,humidity,wind_speed,precipitation]

    return details


def predict_fire(data):
    data = np.array([data])
    pred = model.predict(data)

    if pred.tolist()[0]==0:
        return("No chances of Fire")
    else:
        return("Warning! There a chance of Forest fire")



def main():
    st.set_page_config(page_title="Forest Fire Predictor", layout="wide")
    st.title("🌲 Forest Fire Predictor")
    st.sidebar.header("#####")

    if st.button("Predict"):
        weather_data = fetch_weather_data()
        if weather_data:
            temp = weather_data[0]
            RH = weather_data[1]
            wind = weather_data[2]
            rain = weather_data[3]

            prediction = predict_fire(weather_data)

            st.subheader("Current Weather Details")
            st.write(f"**Temperature:** {temp} °C")
            st.write(f"**Relative Humidity:** {RH}%")
            st.write(f"**Wind Speed:** {wind} km/h")
            st.write(f"**Rainfall:** {rain} mm")

            st.success(f"Prediction: {prediction}")

    st.sidebar.write("Developed by Mukesh")

if __name__ == "__main__":
    main()