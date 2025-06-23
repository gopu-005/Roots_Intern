import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from io import BytesIO
from fpdf import FPDF
import tempfile
import os
from sklearn.linear_model import LinearRegression
import numpy as np

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
    df = df[df['Description'] == product_name].copy()
    df = df.set_index('Date').resample(freq)['qty'].sum().reset_index()
    df.columns = ['ds', 'y']
    df.dropna(inplace=True)

    if len(df) < 2:
        return None

    df['ds_ordinal'] = df['ds'].map(pd.Timestamp.toordinal)
    X = df[['ds_ordinal']]
    y = df['y']

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    mse = np.mean((y - y_pred) ** 2)
    std_dev = np.sqrt(mse)

    last_date = df['ds'].max()
    future_dates = pd.date_range(start=last_date, periods=5, freq=freq)[1:]
    future_ordinal = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    future_preds = model.predict(future_ordinal)

    forecast_df = pd.DataFrame({
        'ds': future_dates,
        'yhat': future_preds,
        'yhat_lower': future_preds - 1.96 * std_dev,
        'yhat_upper': future_preds + 1.96 * std_dev
    })

    df_forecast = pd.concat([df[['ds', 'y']].rename(columns={'y': 'yhat'}), forecast_df], ignore_index=True)
    return model, df_forecast, df[['ds', 'y']]

def generate_pdf_report(product_name, summary, fig):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Sales Forecast Report - {product_name}", ln=True, align='C')
    pdf.ln(10)
    for key, value in summary.items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        fig.savefig(tmpfile.name, format="png", dpi=150, bbox_inches='tight')
        tmpfile_path = tmpfile.name
    pdf.image(tmpfile_path, x=10, y=80, w=190)
    os.remove(tmpfile_path)
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    return BytesIO(pdf_bytes)

# Mobile-responsive CSS
def add_mobile_styles():
    st.markdown("""
    <style>
        /* Mobile-first responsive design */
        @media (max-width: 768px) {
            .main .block-container {
                padding: 1rem 0.5rem !important;
                max-width: 100% !important;
            }
            
            /* Responsive title */
            h1 {
                font-size: 1.5rem !important;
                text-align: center !important;
                margin-bottom: 1rem !important;
            }
            
            /* Mobile sidebar */
            .css-1d391kg {
                width: 100% !important;
            }
            
            /* Responsive metrics */
            .metric-container {
                display: flex !important;
                flex-direction: column !important;
                gap: 0.5rem !important;
            }
            
            /* Full width selectboxes */
            .stSelectbox > div > div {
                width: 100% !important;
            }
            
            /* Responsive text input */
            .stTextInput > div > div > input {
                width: 100% !important;
            }
            
            /* Chart container */
            .chart-container {
                width: 100% !important;
                overflow-x: auto !important;
            }
            
            /* Download button positioning */
            .download-btn {
                width: 100% !important;
                margin: 1rem 0 !important;
            }
            
            /* Responsive dataframe */
            .dataframe {
                font-size: 0.8rem !important;
            }
        }
        
        /* Hover effects for buttons */
        .stDownloadButton > button:hover {
            background-color: #1f77b4 !important;
            color: white !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
        }
        
        .stSelectbox > div > div:hover {
            box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        }
        
        /* Custom metric cards */
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
            margin: 0.5rem 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .metric-value {
            font-size: 1.5rem;
            font-weight: bold;
            margin: 0.5rem 0;
        }
        
        .metric-label {
            font-size: 0.9rem;
            opacity: 0.9;
        }
        
        /* Responsive search bar */
        .search-container {
            position: sticky;
            top: 0;
            background: white;
            padding: 1rem 0;
            z-index: 100;
            border-bottom: 1px solid #e0e0e0;
        }
        
        /* Loading spinner */
        .loading {
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem;
        }
        
        /* Forecast chart styling */
        .forecast-chart {
            background: white;
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

# Check if running on mobile device
def is_mobile():
    # This is a simple check - in real apps you might use user agent detection
    return st.session_state.get('mobile_view', False)

# Streamlit UI
st.set_page_config(
    page_title="Sales Forecasting Dashboard", 
    layout="wide",
    initial_sidebar_state="collapsed"  # Start with sidebar collapsed on mobile
)

# Add mobile styles
add_mobile_styles()

# Mobile-friendly header
st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h1>📊 Sales Forecasting Dashboard</h1>
        <p style="color: #666; margin-top: -1rem;">Analyze and predict sales trends</p>
    </div>
""", unsafe_allow_html=True)

# Load and preprocess data
@st.cache_data
def load_and_process_data():
    df = load_data()
    df = df[df['CategoryCode'].str.lower() != 'vegetables']
    df = get_top_products(df, top_n=75)
    return df

# Loading indicator
with st.spinner('Loading data...'):
    df = load_and_process_data()

# Mobile-optimized sidebar
with st.sidebar:
    st.markdown("### 🔍 Filters")
    
    # Collapsible filter sections
    with st.expander("📂 Category & Location", expanded=True):
        category = st.selectbox(
            "Category", 
            sorted(df['CategoryCode'].dropna().unique()),
            help="Select product category to analyze"
        )
        location = st.selectbox(
            "Location", 
            sorted(df['LocationCode'].dropna().unique()),
            help="Choose store location"
        )
    
    with st.expander("📅 Date Grouping", expanded=True):
        freq_option = st.selectbox(
            "Grouping", 
            ["Monthly", "Quarterly", "Yearly"],
            help="How to group sales data over time"
        )

# Frequency mapping
freq_map = {"Monthly": "MS", "Quarterly": "QS", "Yearly": "YS"}
freq = freq_map[freq_option]

# Filter data
filtered_df = df[(df['CategoryCode'] == category) & (df['LocationCode'] == location)]

# Mobile-optimized metrics display
total_qty = int(filtered_df['qty'].sum())
total_tax = float(filtered_df['TaxableAmount'].sum())
num_products = filtered_df['Description'].nunique()

# Custom metric cards for mobile
st.markdown("""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin: 1rem 0;">
        <div class="metric-card">
            <div class="metric-label">Total Quantity Sold</div>
            <div class="metric-value">{:,}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Total Revenue</div>
            <div class="metric-value">₹{:,.2f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Products Available</div>
            <div class="metric-value">{}</div>
        </div>
    </div>
""".format(total_qty, total_tax, num_products), unsafe_allow_html=True)

# Mobile-optimized search
st.markdown('<div class="search-container">', unsafe_allow_html=True)
search_term = st.text_input(
    "🔍 Search Products", 
    placeholder="Type product name...",
    help="Search for specific products to forecast"
)
st.markdown('</div>', unsafe_allow_html=True)

# Filter products based on search
products = sorted(filtered_df['Description'].unique())
if search_term:
    products = [p for p in products if search_term.lower() in p.lower()]

# Product selection with better mobile UX
if not products:
    st.warning("No products found matching your search criteria.")
else:
    product_selected = st.selectbox(
        "Select Product for Forecasting", 
        products,
        help="Choose a product to generate sales forecast"
    )

    if product_selected:
        # Show loading state
        with st.spinner(f'Generating forecast for {product_selected}...'):
            model_result = forecast_sales(filtered_df, product_selected, freq)
        
        if model_result:
            model, forecast, actual_df = model_result

            # Mobile-optimized chart
            st.markdown('<div class="forecast-chart">', unsafe_allow_html=True)
            
            # Adjust figure size based on screen
            fig_width = 12
            fig_height = 6
            
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            ax.plot(actual_df['ds'], actual_df['y'], label='Actual Sales', marker='o', linewidth=2)
            ax.plot(forecast['ds'], forecast['yhat'], label='Forecasted Sales', linestyle='--', marker='x', linewidth=2)
            ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.2, label='Confidence Interval')
            
            ax.set_title(f"Sales Forecast: {product_selected}", fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Quantity", fontsize=12)
            ax.legend(loc='upper left', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
            
            # Rotate x-axis labels for better mobile readability
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            st.pyplot(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # PDF Download button in top right corner
            col_spacer, col_download = st.columns([8, 2])
            with col_download:
                # Generate PDF report
                summary = {
                    "Product": product_selected,
                    "Total Quantity Sold": f"{actual_df['y'].sum():,.0f}",
                    "Date Range": f"{actual_df['ds'].min().date()} to {actual_df['ds'].max().date()}"
                }
                pdf_data = generate_pdf_report(product_selected, summary, fig)

                st.download_button(
                    label="📄 Download Report",
                    data=pdf_data,
                    file_name=f"{product_selected.replace(' ', '_')}_forecast.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

            # Forecast Summary in center with large font
            st.markdown("""
                <div style="text-align: center; margin: 2rem 0;">
                    <h2 style="font-size: 2rem; color: #333; margin-bottom: 1.5rem;">📊 Forecast Summary</h2>
                </div>
            """, unsafe_allow_html=True)
            
            # Horizontal summary cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                    <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white; margin: 0.5rem;">
                        <div style="font-size: 2rem; font-weight: bold; margin-bottom: 0.5rem;">{len(product_selected.split())}</div>
                        <div style="font-size: 1.1rem; opacity: 0.9;">Product Words</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                    <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); border-radius: 10px; color: white; margin: 0.5rem;">
                        <div style="font-size: 2rem; font-weight: bold; margin-bottom: 0.5rem;">{actual_df['y'].sum():,.0f}</div>
                        <div style="font-size: 1.1rem; opacity: 0.9;">Total Quantity Sold</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                date_range_days = (actual_df['ds'].max() - actual_df['ds'].min()).days
                st.markdown(f"""
                    <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); border-radius: 10px; color: white; margin: 0.5rem;">
                        <div style="font-size: 2rem; font-weight: bold; margin-bottom: 0.5rem;">{date_range_days}</div>
                        <div style="font-size: 1.1rem; opacity: 0.9;">Days of Data</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                    <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); border-radius: 10px; color: white; margin: 0.5rem;">
                        <div style="font-size: 2rem; font-weight: bold; margin-bottom: 0.5rem;">{freq_option[0]}</div>
                        <div style="font-size: 1.1rem; opacity: 0.9;">Forecast Type</div>
                    </div>
                """, unsafe_allow_html=True)

            # Business Intelligence Suggestions
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Calculate trend and generate suggestions
            recent_sales = actual_df['y'].tail(3).mean()
            overall_avg = actual_df['y'].mean()
            future_forecast_avg = forecast[forecast['ds'] > actual_df['ds'].max()]['yhat'].mean()
            
            # Determine trend
            if future_forecast_avg > recent_sales * 1.1:
                trend = "increasing"
                trend_icon = "📈"
                trend_color = "#28a745"
                suggestion = "**INCREASE STOCK** - Forecast shows strong upward trend. Consider increasing inventory by 15-20% to meet growing demand."
                action = "Scale up manufacturing and ensure adequate raw materials."
            elif future_forecast_avg < recent_sales * 0.9:
                trend = "decreasing"
                trend_icon = "📉"
                trend_color = "#dc3545"
                suggestion = "**REDUCE STOCK** - Forecast indicates declining demand. Consider reducing inventory by 10-15% to avoid overstocking."
                action = "Optimize production schedule and consider promotional strategies."
            else:
                trend = "stable"
                trend_icon = "📊"
                trend_color = "#ffc107"
                suggestion = "**MAINTAIN CURRENT LEVELS** - Forecast shows stable demand. Continue with current inventory management."
                action = "Monitor market trends and maintain steady production."

            # Display suggestions
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, {trend_color}15 0%, {trend_color}05 100%); 
                     border-left: 5px solid {trend_color}; padding: 2rem; border-radius: 10px; margin: 2rem 0;">
                    <h3 style="color: {trend_color}; font-size: 1.8rem; margin-bottom: 1rem;">
                        {trend_icon} Business Intelligence Recommendations
                    </h3>
                    <div style="font-size: 1.3rem; color: #333; margin-bottom: 1rem; font-weight: 600;">
                        {suggestion}
                    </div>
                    <div style="font-size: 1.1rem; color: #666; margin-bottom: 1rem;">
                        <strong>Manufacturing Guidance:</strong> {action}
                    </div>
                    <div style="font-size: 1rem; color: #888; border-top: 1px solid #eee; padding-top: 1rem;">
                        <strong>Trend Analysis:</strong> Future forecast average: {future_forecast_avg:.1f} units 
                        vs Recent average: {recent_sales:.1f} units ({((future_forecast_avg/recent_sales - 1) * 100):+.1f}% change)
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.error("❌ Not enough data to generate forecast for this product.")
            st.info("💡 Try selecting a product with more historical sales data.")

# Mobile-optimized data viewer
with st.expander("📋 View Raw Data", expanded=False):
    st.markdown("### Filtered Dataset Preview")
    
    # Show data in mobile-friendly format
    display_cols = ['Date', 'Description', 'qty', 'TaxableAmount']
    mobile_df = filtered_df[display_cols].head(50)
    
    st.dataframe(
        mobile_df,
        use_container_width=True,
        hide_index=True
    )
    
    if len(filtered_df) > 50:
        st.info(f"Showing first 50 of {len(filtered_df)} total records.")

# Footer for mobile
st.markdown("""
    <div style="text-align: center; padding: 2rem 0; color: #666; border-top: 1px solid #e0e0e0; margin-top: 2rem;">
        <p>📱 Mobile-optimized Sales Forecasting Dashboard</p>
        <p style="font-size: 0.8rem;">Swipe left to access filters • Tap charts to zoom</p>
    </div>
""", unsafe_allow_html=True)