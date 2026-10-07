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

    st.write("""
    This page explores factors that may influence Airbnb prices in New York City.
    We focus on location, property type, minimum-night requirements, and
    relationships between numerical variables.
    """)

    # Clean data for visualizations
    viz_df = df.dropna(subset=["Price"]).copy()

    # -------------------------------
    # 1. Average Price by Borough
    # -------------------------------

    st.subheader("🏙️ Average Price by Borough")

    borough_price = (
        viz_df.groupby("Boroughs")["Price"]
        .mean()
        .sort_values(ascending=False)
    )

    fig1, ax1 = plt.subplots()

    sns.barplot(
        x=borough_price.index,
        y=borough_price.values,
        ax=ax1
    )

    ax1.set_xlabel("Borough")
    ax1.set_ylabel("Average Price ($)")
    ax1.set_title("Average Airbnb Price by NYC Borough")
    plt.xticks(rotation=45)

    st.pyplot(fig1)

st.write("""
**Insight:** Location appears to have a strong relationship with Airbnb pricing.
Manhattan has the highest average nightly price at approximately $178, followed
by Brooklyn at approximately $120. The Bronx has the lowest average price at
around $80. This suggests that borough should be considered when predicting
the price of an Airbnb listing.
""")

    # -------------------------------
    # 2. Average Price by Property Type
    # -------------------------------

    st.subheader("🏠 Average Price by Property Type")

    property_price = (
        viz_df.groupby("Prop_Type")["Price"]
        .mean()
        .sort_values(ascending=False)
    )

    fig2, ax2 = plt.subplots()

    sns.barplot(
        x=property_price.index,
        y=property_price.values,
        ax=ax2
    )

    ax2.set_xlabel("Property Type")
    ax2.set_ylabel("Average Price ($)")
    ax2.set_title("Average Airbnb Price by Property Type")
    plt.xticks(rotation=45)

    st.pyplot(fig2)

    st.write("""
    Property type may be an important predictor of Airbnb price.
    This visualization compares the average nightly price for the
    different types of properties in the dataset.
    """)

    # -------------------------------
    # 3. Price vs Minimum Nights
    # -------------------------------

    st.subheader("🌙 Price vs. Minimum Nights")

   scatter_df = viz_df.dropna(subset=["Min_Nights"]).copy()

    scatter_df = scatter_df[
        (scatter_df["Price"] <= 1000) &
        (scatter_df["Min_Nights"] <= 60)
    ]

    fig3, ax3 = plt.subplots()

    sns.scatterplot(
        data=scatter_df,
        x="Min_Nights",
        y="Price",
        alpha=0.4,
        ax=ax3
    )

    ax3.set_xlabel("Minimum Nights")
    ax3.set_ylabel("Price ($)")
    ax3.set_title("Airbnb Price vs. Minimum Nights")

    st.pyplot(fig3)

    st.write("""
    This scatter plot examines whether listings with different
    minimum-night requirements tend to have different prices.
    """)

    # -------------------------------
    # 4. Correlation Heatmap
    # -------------------------------

    st.subheader("🔥 Correlation Heatmap")

    numeric_df = viz_df.select_dtypes(include=np.number)

    fig4, ax4 = plt.subplots(figsize=(10, 7))

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax4
    )

    ax4.set_title("Correlation Between Numerical Variables")

    st.pyplot(fig4)

    st.write("""
    The correlation matrix helps identify numerical variables that
    have relationships with Airbnb price. Variables with stronger
    correlations may be useful predictors in our linear regression model.
    """)

elif page == "Price Prediction 🤖":

    st.title("🤖 Airbnb Price Prediction")

    st.info("We'll build our linear regression model here next!")
