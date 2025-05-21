import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.title("Flexible AI-Driven Stock Price Prediction App")

uploaded_file = st.file_uploader("Upload your stock CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Columns Found in Dataset")
    st.write(list(df.columns))

    date_col = st.selectbox("Select Date column", options=df.columns)
    price_col = st.selectbox("Select Price column", options=df.columns)

    try:
        # Preprocess
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        df = df.dropna(subset=[date_col, price_col])
        df = df.reset_index(drop=True)

        st.subheader("Dataset Preview after Preprocessing")
        st.write(df.head())

        # Visualization
        st.subheader(f"{price_col} Over Time")
        fig, ax = plt.subplots()
        ax.plot(df[date_col], df[price_col], label=price_col, color='blue')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

        # Feature Engineering
        df['Days'] = (df[date_col] - df[date_col].min()).dt.days
        X = df[['Days']]
        y = df[price_col]

        # Model Training
        test_size = st.slider("Test Data Size (%)", min_value=10, max_value=50, value=20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, shuffle=False)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluation
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.subheader("Model Evaluation")
        st.write(f"Mean Squared Error: {mse:.2f}")
        st.write(f"R² Score: {r2:.4f}")

        # Prediction Visualization
        st.subheader("Actual vs Predicted Prices")
        fig2, ax2 = plt.subplots()
        ax2.plot(df[date_col].iloc[-len(y_test):], y_test, label='Actual', color='blue')
        ax2.plot(df[date_col].iloc[-len(y_test):], y_pred, label='Predicted', color='red', linestyle='--')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Price')
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"Error processing data: {e}")

else:
    st.info("Please upload a CSV file to start.")