import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from io import BytesIO
from fpdf import FPDF
import tempfile
import os

# Load and clean the data
def load_data():
    df = pd.read_excel("new dataset.xlsx", skiprows=1)
    df = df.rename(columns={
        'Unnamed: 2': 'Date',
        'Unnamed: 3': 'Description',
        'Unnamed: 4': 'CategoryCode',
        'Unnamed: 5': 'DepartmentCode',
        'Unnamed: 6': 'LocationCode',
        'Unnamed: 7': 'qty',
        'Unnamed: 8': 'TaxableAmount'
    })
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def get_top_products(df, top_n=75):
    top_products = df.groupby('Description')['qty'].sum().sort_values(ascending=False).head(top_n).index.tolist()
    return df[df['Description'].isin(top_products)]

def forecast_sales(df, product_name, freq):
    product_df = df[df['Description'] == product_name].copy()
    product_df['Date'] = pd.to_datetime(product_df['Date'])
    product_df.set_index('Date', inplace=True)
    resampled = product_df['qty'].resample(freq).sum().reset_index()
    resampled.columns = ['ds', 'y']
    if len(resampled) < 2:
        return None
    model = Prophet()
    model.fit(resampled)
    future = model.make_future_dataframe(periods=4, freq=freq)
    forecast = model.predict(future)
    return model, forecast, resampled

def generate_pdf_report(product_name, summary, fig):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt=f"Sales Forecast Report - {product_name}", ln=True, align='C')
    pdf.ln(10)

    for key, value in summary.items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        fig.savefig(tmpfile.name, format="png")
        tmpfile_path = tmpfile.name

    pdf.image(tmpfile_path, x=10, y=80, w=190)
    os.remove(tmpfile_path)

    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    return BytesIO(pdf_bytes)

# Streamlit UI
st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")
st.title("📊 Sales Forecasting Dashboard")

# Load and preprocess data
df = load_data()
df = get_top_products(df, top_n=75)

# Sidebar filters
st.sidebar.header("🔍 Filter Options")
category = st.sidebar.selectbox("Select Category", sorted(df['CategoryCode'].dropna().unique()))
freq_option = st.sidebar.selectbox("Select Date Grouping", ["Monthly", "Quarterly", "Yearly"])
location = st.sidebar.selectbox("Select Location", sorted(df['LocationCode'].dropna().unique()))

# Frequency mapping
freq_map = {"Monthly": "MS", "Quarterly": "QS", "Yearly": "YS"}
freq = freq_map[freq_option]

# Filter data
filtered_df = df[(df['CategoryCode'] == category) & (df['LocationCode'] == location)]

# Total Sales Summary
total_qty = int(filtered_df['qty'].sum())
total_tax = float(filtered_df['TaxableAmount'].sum())
num_products = filtered_df['Description'].nunique()

st.markdown("### 🧾 Sales Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Total Quantity Sold", f"{total_qty:,}")
col2.metric("Total Taxable Amount", f"₹{total_tax:,.2f}")
col3.metric("Products Forecasted", num_products)

# Download placeholder
download_col = st.columns([1, 8])[0]
download_placeholder = download_col.empty()

# Search bar for products
search_term = st.text_input("🔍 Search Product Name")
products = sorted(filtered_df['Description'].unique())
if search_term:
    products = [p for p in products if search_term.lower() in p.lower()]

product_selected = st.selectbox("Select Product for Forecasting", products)

if product_selected:
    model_result = forecast_sales(filtered_df, product_selected, freq)
    if model_result:
        model, forecast, actual_df = model_result

        # Plot Actual vs Forecast
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(actual_df['ds'], actual_df['y'], label='Actual Sales', marker='o')
        ax.plot(forecast['ds'], forecast['yhat'], label='Forecasted Sales', linestyle='--')
        ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.2, label='Confidence Interval')
        ax.set_title(f"Sales Forecast for {product_selected}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Quantity")
        ax.legend()
        ax.grid(True)
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
        st.pyplot(fig)

        # Summary for PDF
        summary = {
            "Product": product_selected,
            "Total Quantity Sold": f"{actual_df['y'].sum():,.0f}",
            "Date Range": f"{actual_df['ds'].min().date()} to {actual_df['ds'].max().date()}"
        }
        pdf_data = generate_pdf_report(product_selected, summary, fig)

        # Display download button at the top left with hover effect
        download_placeholder.download_button(
            label="📄 Download PDF Report",
            data=pdf_data,
            file_name=f"{product_selected}_forecast.pdf",
            mime="application/pdf",
            key="download_pdf"
        )
    else:
        st.warning("Not enough data to generate forecast for this product.")

# Show raw data
with st.expander("🔎 See Filtered Raw Data"):
    st.dataframe(filtered_df.head(100))

# Custom CSS for hover effect on download button
st.markdown("""
    <style>
        button[kind="primary"]:hover {
            background-color: #1f77b4 !important;
            color: white !important;
        }
    </style>
    """, unsafe_allow_html=True)
