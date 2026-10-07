import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =====================================
# PAGE SETUP
# =====================================

st.set_page_config(
    page_title="NYC Airbnb Dashboard",
    page_icon="🏙️",
    layout="centered"
)

# Load dataset
df = pd.read_csv("Airbnb_NYC.csv")


# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("🏙️ NYC Airbnb")

page = st.sidebar.selectbox(
    "Select Page",
    [
        "Business Case 📘",
        "Data Visualization 📊",
        "Price Prediction 🤖"
    ]
)


# =====================================
# PAGE 1 — BUSINESS CASE
# =====================================

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

    rows = st.slider(
        "Number of rows to display",
        5,
        20,
        5
    )

    st.dataframe(df.head(rows))

    st.subheader("Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Number of Listings",
            len(df)
        )

    with col2:
        st.metric(
            "Number of Variables",
            len(df.columns)
        )

    st.subheader("Missing Values")

    st.dataframe(
        df.isnull().sum().to_frame("Missing Values")
    )


# =====================================
# PAGE 2 — DATA VISUALIZATION
# =====================================

elif page == "Data Visualization 📊":

    st.title("📊 NYC Airbnb Data Visualization")

    st.write("""
    This page explores factors that may influence Airbnb prices in New York City.
    We focus on location, property type, minimum-night requirements, and
    relationships between numerical variables.
    """)

    # Remove rows where price is missing
    viz_df = df.dropna(subset=["Price"]).copy()


    # ---------------------------------
    # 1. Average Price by Borough
    # ---------------------------------

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
    **Insight:** Location appears to have a strong relationship with Airbnb
    pricing. Manhattan has the highest average nightly price at approximately
    $178, followed by Brooklyn at approximately $120. The Bronx has the lowest
    average price at around $80. This suggests that borough should be considered
    when predicting the price of an Airbnb listing.
    """)


    # ---------------------------------
    # 2. Average Price by Property Type
    # ---------------------------------

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
    **Insight:** Property type shows a substantial difference in average price.
    Entire homes have an average nightly price of approximately $193, compared
    with about $84 for private rooms and $63 for shared rooms. This suggests
    that property type may be an important predictor of Airbnb price.
    """)


    # ---------------------------------
    # 3. Price vs Minimum Nights
    # ---------------------------------

    st.subheader("🌙 Price vs. Minimum Nights")

    scatter_df = (
        viz_df
        .dropna(subset=["Min_Nights"])
        .copy()
    )

    # Remove extreme values for clearer visualization
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
    **Insight:** Most listings are concentrated at relatively low minimum-night
    requirements. The relationship between minimum nights and price does not
    appear as strong as the differences observed for borough and property type.
    """)


    # ---------------------------------
    # 4. Correlation Heatmap
    # ---------------------------------

    st.subheader("🔥 Correlation Heatmap")

    numeric_df = viz_df.select_dtypes(
        include=np.number
    )

    fig4, ax4 = plt.subplots(
        figsize=(10, 7)
    )

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax4
    )

    ax4.set_title(
        "Correlation Between Numerical Variables"
    )

    st.pyplot(fig4)

    st.write("""
    **Insight:** The heatmap shows that most numerical variables have relatively
    weak linear relationships with price. This suggests that categorical
    variables such as borough and property type may provide important
    additional information for predicting Airbnb prices.
    """)


# =====================================
# PAGE 3 — PRICE PREDICTION
# =====================================

elif page == "Price Prediction 🤖":

    st.title("🤖 NYC Airbnb Price Prediction")

    st.write("""
    This model uses linear regression to estimate the nightly price of an
    Airbnb listing based on its location, property type, and listing
    characteristics.
    """)


    # ---------------------------------
    # 1. Prepare Data
    # ---------------------------------

    model_df = df[
        [
            "Boroughs",
            "Prop_Type",
            "Min_Nights",
            "Host_Listing_Cnt",
            "Days_Available",
            "Review_Cnt",
            "Reviews30d",
            "Price"
        ]
    ].copy()

    # Remove missing values
    model_df = model_df.dropna()

    # Remove extreme price outliers
    model_df = model_df[
        (model_df["Price"] > 0) &
        (model_df["Price"] <= 1000)
    ]


    # ---------------------------------
    # 2. Define X and Y
    # ---------------------------------

    X = model_df[
        [
            "Boroughs",
            "Prop_Type",
            "Min_Nights",
            "Host_Listing_Cnt",
            "Days_Available",
            "Review_Cnt",
            "Reviews30d"
        ]
    ]

    y = model_df["Price"]

    # Convert categorical variables
    # into numerical dummy variables
    X = pd.get_dummies(
        X,
        columns=[
            "Boroughs",
            "Prop_Type"
        ],
        drop_first=True
    )


    # ---------------------------------
    # 3. Train/Test Split
    # ---------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )


    # ---------------------------------
    # 4. Train Model
    # ---------------------------------

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )


    # ---------------------------------
    # 5. Model Evaluation
    # ---------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    st.subheader("📊 Model Performance")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "MAE",
            f"${mae:,.2f}"
        )

    with col2:
        st.metric(
            "MSE",
            f"{mse:,.2f}"
        )

    with col3:
        st.metric(
            "R² Score",
            f"{r2:.3f}"
        )

    st.caption("""
    MAE represents the model's average prediction error in dollars.
    R² measures how much of the variation in Airbnb prices is explained
    by the model.
    """)


    # ---------------------------------
    # 6. Actual vs Predicted
    # ---------------------------------

    st.subheader("📈 Actual vs. Predicted Prices")

    fig5, ax5 = plt.subplots()

    ax5.scatter(
        y_test,
        predictions,
        alpha=0.3
    )

    min_value = min(
        y_test.min(),
        predictions.min()
    )

    max_value = max(
        y_test.max(),
        predictions.max()
    )

    ax5.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--"
    )

    ax5.set_xlabel(
        "Actual Price ($)"
    )

    ax5.set_ylabel(
        "Predicted Price ($)"
    )

    ax5.set_title(
        "Actual vs. Predicted Airbnb Prices"
    )

    st.pyplot(fig5)

    st.write("""
    Points closer to the diagonal line represent more accurate predictions.
    Larger distances from the line indicate greater prediction errors.
    """)


    # ---------------------------------
    # 7. Interactive Predictor
    # ---------------------------------

    st.divider()

    st.subheader("🔮 Predict an Airbnb Price")

    st.write("""
    Enter the characteristics of an Airbnb listing below to estimate
    its nightly price.
    """)

    borough = st.selectbox(
        "Borough",
        sorted(
            model_df["Boroughs"].unique()
        )
    )

    property_type = st.selectbox(
        "Property Type",
        sorted(
            model_df["Prop_Type"].unique()
        )
    )

    minimum_nights = st.number_input(
        "Minimum Nights",
        min_value=1,
        value=2,
        step=1
    )

    host_listing_count = st.number_input(
        "Number of Listings Owned by Host",
        min_value=1,
        value=1,
        step=1
    )

    days_available = st.slider(
        "Days Available per Year",
        min_value=0,
        max_value=365,
        value=180
    )

    review_count = st.number_input(
        "Total Number of Reviews",
        min_value=0,
        value=20,
        step=1
    )

    reviews_30d = st.number_input(
        "Reviews in the Last 30 Days",
        min_value=0,
        value=1,
        step=1
    )


    # ---------------------------------
    # 8. Make Prediction
    # ---------------------------------

    if st.button(
        "🔮 Predict Nightly Price",
        type="primary"
    ):

        new_listing = pd.DataFrame(
            {
                "Boroughs": [borough],
                "Prop_Type": [property_type],
                "Min_Nights": [minimum_nights],
                "Host_Listing_Cnt": [host_listing_count],
                "Days_Available": [days_available],
                "Review_Cnt": [review_count],
                "Reviews30d": [reviews_30d]
            }
        )

        # Encode categorical variables
        new_listing = pd.get_dummies(
            new_listing,
            columns=[
                "Boroughs",
                "Prop_Type"
            ]
        )

        # Make columns match training data
        new_listing = new_listing.reindex(
            columns=X.columns,
            fill_value=0
        )

        predicted_price = model.predict(
            new_listing
        )[0]

        # Avoid displaying negative prices
        predicted_price = max(
            predicted_price,
            0
        )

        st.success(
            f"Estimated Nightly Price: ${predicted_price:,.2f}"
        )

        st.write(
            f"Based on the model, a **{property_type}** in "
            f"**{borough}** with these characteristics has an estimated "
            f"nightly price of **${predicted_price:,.2f}**."
        )
