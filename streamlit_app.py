import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Page setup
st.set_page_config(
    page_title="NYC Airbnb Dashboard",
    page_icon="🏙️",
    layout="centered"
)

# Load dataset
df = pd.read_csv("Airbnb_NYC.csv")

# Sidebar
st.sidebar.title("🏙️ NYC Airbnb")
page = st.sidebar.selectbox(
    "Select Page",
    ["Business Case 📘", "Data Visualization 📊", "Price Prediction 🤖"]
)

# Business Case Page
if page == "Business Case 📘":

    st.title("🏙️ NYC Airbnb Price Prediction")

    st.subheader("🎯 Business Problem")
    st.write("""
    Airbnb hosts need to determine an appropriate nightly price for their
    properties. Pricing a listing too high may reduce demand, while pricing
    too low may result in lost potential revenue.
    """)

    st.subheader("🔎 Objective")
    st.write("""
    Our goal is to analyze Airbnb listings in New York City, identify factors
    associated with listing prices, and build a linear regression model that
    can predict the nightly price of an Airbnb.
    """)

    st.subheader("📂 Dataset Preview")

    rows = st.slider("Number of rows to display", 5, 20, 5)

    st.dataframe(df.head(rows))

    st.subheader("Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Number of Listings", len(df))

    with col2:
        st.metric("Number of Variables", len(df.columns))

    st.subheader("Missing Values")
    st.dataframe(df.isnull().sum().to_frame("Missing Values"))

elif page == "Data Visualization 📊":

    st.title("📊 NYC Airbnb Data Visualization")

    st.info("We'll build the visualizations here next!")

elif page == "Price Prediction 🤖":

    st.title("🤖 Airbnb Price Prediction")

    st.info("We'll build our linear regression model here next!")
