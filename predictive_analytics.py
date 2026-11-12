
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Predictive Analytics Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Predictive Analytics Dashboard")
st.write("Forecast future sales using historical data and Machine Learning.")

# --------------------------------
# Generate Sample Historical Data
# --------------------------------

def generate_sample_data():
    np.random.seed(42)

    dates = pd.date_range(
        start="2024-01-01",
        periods=36,
        freq="MS"
    )

    sales = [
        50000 + (i * 1800) +
        8000 * np.sin(i * 2 * np.pi / 12) +
        np.random.randint(-3000, 3000)
        for i in range(36)
    ]

    return pd.DataFrame({
        "Date": dates,
        "Sales": sales
    })


# --------------------------------
# Upload Historical Dataset
# --------------------------------

st.sidebar.header("Upload Historical Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Unable to read file: {e}")
        st.stop()
else:
    df = generate_sample_data()
    st.sidebar.info("Using sample historical sales data.")

# --------------------------------
# Clean and Preprocess Data
# --------------------------------

df.columns = df.columns.str.strip()

required_columns = ["Date", "Sales"]

if not all(col in df.columns for col in required_columns):
    st.error("Your dataset must contain Date and Sales columns.")
    st.stop()

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")

df = df.dropna(subset=["Date", "Sales"])

# Sort data by date
df = df.sort_values("Date")

# Group sales by month
df = (
    df.set_index("Date")
    .resample("MS")["Sales"]
    .sum()
    .reset_index()
)

# Remove missing values
df = df.dropna()

if len(df) < 6:
    st.error("Please upload at least 6 months of historical sales data.")
    st.stop()

# --------------------------------
# Forecast Settings
# --------------------------------

st.sidebar.header("Prediction Settings")

forecast_months = st.sidebar.slider(
    "Number of Months to Forecast",
    min_value=1,
    max_value=12,
    value=6
)

# --------------------------------
# Prepare Data for Machine Learning
# --------------------------------

df["Time"] = np.arange(len(df))

X = df[["Time"]]
y = df["Sales"]

# Split historical data into training and testing sets
split_index = max(3, int(len(df) * 0.8))

if split_index >= len(df):
    split_index = len(df) - 1

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# --------------------------------
# Train and Evaluate Model
# --------------------------------

test_model = LinearRegression()

test_model.fit(X_train, y_train)

test_predictions = test_model.predict(X_test)

mae = mean_absolute_error(y_test, test_predictions)

rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

r2 = r2_score(y_test, test_predictions) if len(y_test) >= 2 else np.nan

# --------------------------------
# Train Final Model
# --------------------------------

model = LinearRegression()

model.fit(X, y)

# Predict future sales
future_time = np.arange(
    len(df),
    len(df) + forecast_months
)

future_X = pd.DataFrame({
    "Time": future_time
})

future_sales = model.predict(future_X)

future_dates = pd.date_range(
    start=df["Date"].iloc[-1] + pd.offsets.MonthBegin(1),
    periods=forecast_months,
    freq="MS"
)

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted_Sales": future_sales
})

# --------------------------------
# Display Key Performance Indicators
# --------------------------------

st.subheader("Business Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Historical Sales",
    f"₹{df['Sales'].sum():,.0f}"
)

col2.metric(
    "Average Monthly Sales",
    f"₹{df['Sales'].mean():,.0f}"
)

col3.metric(
    "Mean Absolute Error",
    f"₹{mae:,.0f}"
)

col4.metric(
    "Model R² Score",
    f"{r2:.2f}" if not np.isnan(r2) else "N/A"
)

st.divider()

# --------------------------------
# Historical Sales Visualization
# --------------------------------

st.subheader("Historical Sales Trends")

fig1 = px.line(
    df,
    x="Date",
    y="Sales",
    markers=True,
    title="Monthly Historical Sales"
)

fig1.update_layout(
    xaxis_title="Date",
    yaxis_title="Sales (₹)"
)

st.plotly_chart(fig1, width="stretch")

# --------------------------------
# Actual vs Predicted Visualization
# --------------------------------

st.subheader("Actual vs Predicted Sales")

comparison_df = pd.DataFrame({
    "Date": df["Date"].iloc[split_index:].values,
    "Actual Sales": y_test.values,
    "Predicted Sales": test_predictions
})

fig2 = px.line(
    comparison_df,
    x="Date",
    y=["Actual Sales", "Predicted Sales"],
    markers=True,
    title="Model Performance on Test Data"
)

st.plotly_chart(fig2, width="stretch")

# --------------------------------
# Future Sales Forecast
# --------------------------------

st.subheader("Future Sales Prediction")

fig3 = px.line(
    forecast_df,
    x="Date",
    y="Predicted_Sales",
    markers=True,
    title="Future Monthly Sales Forecast"
)

st.plotly_chart(fig3, width="stretch")

st.write("### Forecast Results")

st.dataframe(
    forecast_df,
    width="stretch",
    hide_index=True
)

# --------------------------------
# Model Evaluation
# --------------------------------

st.subheader("Model Evaluation")

col1, col2 = st.columns(2)

with col1:
    st.write("**Mean Absolute Error (MAE)**")
    st.write(f"₹{mae:,.2f}")
    st.caption("Average absolute prediction error.")

    st.write("**Root Mean Squared Error (RMSE)**")
    st.write(f"₹{rmse:,.2f}")
    st.caption("Measures prediction errors, giving larger errors more weight.")

with col2:
    st.write("**R² Score**")
    st.write(f"{r2:.4f}" if not np.isnan(r2) else "Not available")
    st.caption("Measures how well the model explains variation in test sales.")

# --------------------------------
# Download Forecast Report
# --------------------------------

download_df = forecast_df.copy()

csv_data = download_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Sales Forecast Report",
    data=csv_data,
    file_name="sales_forecast_report.csv",
    mime="text/csv"
)

st.success("Sales forecasting analysis completed!")
