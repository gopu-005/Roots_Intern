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

# Custom CSS for better styling
def load_css():
    st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    
    /* Title styling */
    .main-title {
        color: white;
        text-align: center;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Card styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 1rem 0;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
    }
    
    .metric-title {
        font-size: 1.1rem;
        color: #2c3e50;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #3498db;
        margin: 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
    
    .css-1d391kg .css-1v0mbdj {
        color: white;
    }
    
    /* Sidebar text styling */
    .css-1d391kg .stSelectbox label {
        color: white !important;
        font-weight: 600;
    }
    
    .css-1d391kg .stSlider label {
        color: white !important;
        font-weight: 600;
    }
    
    .css-1d391kg .stMarkdown {
        color: white !important;
    }
    
    /* Select box styling */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        border: 2px solid #3498db;
    }
    
    /* Select box text */
    .stSelectbox > div > div > div {
        color: #2c3e50 !important;
        font-weight: 600;
    }
    
    /* Main selectbox styling */
    div[data-testid="stSelectbox"] > div > div {
        background: rgba(255, 255, 255, 0.95);
        color: #2c3e50 !important;
        border-radius: 10px;
        border: 2px solid #3498db;
    }
    
    /* Sidebar toggle button */
    .css-1544g2n {
        background: #3498db;
        color: white;
        border-radius: 50%;
        padding: 0.5rem;
    }
    
    /* Hamburger menu styling */
    .sidebar-toggle {
        position: fixed;
        top: 1rem;
        left: 1rem;
        z-index: 999;
        background: #3498db;
        color: white;
        border: none;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        font-size: 20px;
        cursor: pointer;
        box-shadow: 0 4px 12px rgba(52, 152, 219, 0.3);
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .sidebar-toggle:hover {
        background: #2980b9;
        transform: scale(1.1);
    }
    
    /* Recommendation card styling */
    .recommendation-card {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border-left: 5px solid;
    }
    
    .recommendation-positive {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left-color: #28a745;
        color: #155724;
    }
    
    .recommendation-negative {
        background: linear-gradient(135deg, #f8d7da 0%, #f1c2c7 100%);
        border-left-color: #dc3545;
        color: #721c24;
    }
    
    .recommendation-neutral {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border-left-color: #ffc107;
        color: #856404;
    }
    
    .recommendation-title {
        font-size: 1.3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .recommendation-text {
        font-size: 1rem;
        line-height: 1.6;
    }
    
    /* Chart container */
    .chart-container {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin: 1.5rem 0;
    }
    
    /* Download button styling */
    .stDownloadButton > button {
        background: linear-gradient(45deg, #3498db, #2980b9);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton > button:hover {
        background: linear-gradient(45deg, #2980b9, #3498db);
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(52, 152, 219, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# Load and clean the data
@st.cache_data
def load_data():
    df = pd.read_excel("veyr dataset.xlsx")
    df.columns = [col.strip() for col in df.columns]
    
    if 'Date' not in df.columns:
        df = df.rename(columns={
            df.columns[2]: 'Date',
            df.columns[3]: 'Description',
            df.columns[4]: 'CategoryCode',
            df.columns[5]: 'DepartmentCode',
            df.columns[6]: 'LocationCode',
            df.columns[7]: 'qty',
            df.columns[8]: 'TaxableAmount'
        })

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Date'], inplace=True)
    return df

def forecast_sales(df, product_name, freq):
    """Forecast sales for a specific product or all products combined"""
    if product_name == 'ALL':
        # Aggregate all products
        df_agg = df.copy()
        df_agg = df_agg.set_index('Date').resample(freq)['qty'].sum().reset_index()
        df_agg.columns = ['ds', 'y']
    else:
        df_agg = df[df['Description'] == product_name].copy()
        df_agg = df_agg.set_index('Date').resample(freq)['qty'].sum().reset_index()
        df_agg.columns = ['ds', 'y']
    
    df_agg.dropna(inplace=True)

    if len(df_agg) < 2:
        return None

    df_agg['ds_ordinal'] = df_agg['ds'].map(pd.Timestamp.toordinal)
    X = df_agg[['ds_ordinal']]
    y = df_agg['y']

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    mse = np.mean((y - y_pred) ** 2)
    std_dev = np.sqrt(mse)

    last_date = df_agg['ds'].max()
    future_dates = pd.date_range(start=last_date, periods=5, freq=freq)[1:]
    future_ordinal = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    future_preds = model.predict(future_ordinal)

    forecast_df = pd.DataFrame({
        'ds': future_dates,
        'yhat': future_preds,
        'yhat_lower': future_preds - 1.96 * std_dev,
        'yhat_upper': future_preds + 1.96 * std_dev
    })

    df_forecast = pd.concat([df_agg[['ds', 'y']].rename(columns={'y': 'yhat'}), forecast_df], ignore_index=True)
    return model, df_forecast, df_agg[['ds', 'y']]

def calculate_category_trend(df, category, freq):
    """Calculate category trend performance"""
    if category == 'ALL':
        category_df = df.copy()
    else:
        category_df = df[df['CategoryCode'] == category].copy()
    
    if category_df.empty:
        return 0, "⚠️", "#ffc107"
    
    # Group by frequency to get trend data
    category_df = category_df.set_index('Date').resample(freq)['qty'].sum().reset_index()
    category_df.columns = ['Date', 'qty']
    category_df.dropna(inplace=True)
    
    if len(category_df) < 2:
        return 0, "⚠️", "#ffc107"
    
    # Calculate trend
    recent_sales = category_df['qty'].iloc[-min(3, len(category_df)):].mean()
    earlier_sales = category_df['qty'].iloc[:-min(3, len(category_df))].mean() if len(category_df) > 3 else category_df['qty'].iloc[0]
    
    trend_change = ((recent_sales - earlier_sales) / earlier_sales * 100) if earlier_sales > 0 else 0
    
    # Determine symbol and color based on trend
    if trend_change > 15:
        return trend_change, "▲", "#27ae60"  # Green triangle up
    elif trend_change < -15:
        return trend_change, "▼", "#e74c3c"  # Red triangle down
    else:
        return trend_change, "◆", "#f39c12"  # Yellow diamond for neutral

def analyze_trend_and_recommend(actual_df, forecast_df):
    """Analyze sales trend and provide stock recommendations"""
    if len(actual_df) < 2:
        return "neutral", "Insufficient data for analysis", "⚠️"
    
    # Calculate trend
    recent_sales = actual_df['y'].iloc[-min(3, len(actual_df)):].mean()
    earlier_sales = actual_df['y'].iloc[:-min(3, len(actual_df))].mean() if len(actual_df) > 3 else actual_df['y'].iloc[0]
    
    trend_change = ((recent_sales - earlier_sales) / earlier_sales * 100) if earlier_sales > 0 else 0
    
    # Get future forecast average
    future_forecast = forecast_df[forecast_df['ds'] > actual_df['ds'].max()]
    avg_future_sales = future_forecast['yhat'].mean() if not future_forecast.empty else 0
    
    # Recommendation logic
    if trend_change > 20 and avg_future_sales > recent_sales:
        return "positive", f"📈 **STRONG BUY SIGNAL**: Sales trending upward by {trend_change:.1f}%. Forecast shows continued growth. Recommended action: **Increase stock by 25-30%**. Expected future sales: {avg_future_sales:.0f} units.", "🟢"
    elif trend_change > 10 and avg_future_sales > recent_sales * 0.8:
        return "positive", f"📊 **MODERATE BUY**: Sales growing by {trend_change:.1f}%. Forecast indicates stable demand. Recommended action: **Maintain current stock levels with 15% buffer**. Expected future sales: {avg_future_sales:.0f} units.", "🟢"
    elif trend_change < -20 or avg_future_sales < recent_sales * 0.6:
        return "negative", f"📉 **CAUTION**: Sales declining by {abs(trend_change):.1f}%. Forecast shows continued decline. Recommended action: **Reduce stock by 20-25%** to avoid overstock. Expected future sales: {avg_future_sales:.0f} units.", "🔴"
    elif trend_change < -10:
        return "negative", f"⚠️ **MONITOR CLOSELY**: Sales declining by {abs(trend_change):.1f}%. Forecast shows potential stabilization. Recommended action: **Reduce stock by 10-15%** and monitor weekly. Expected future sales: {avg_future_sales:.0f} units.", "🟡"
    else:
        return "neutral", f"📊 **STABLE DEMAND**: Sales relatively stable with {trend_change:.1f}% change. Forecast shows consistent demand. Recommended action: **Maintain current stock levels**. Expected future sales: {avg_future_sales:.0f} units.", "🟡"

def clean_text_for_pdf(text):
    """Remove emojis and special characters that can't be encoded in latin-1"""
    import re
    # Remove emojis and special characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def get_product_analysis(df, product_name, freq):
    """Get detailed analysis for a single product"""
    if product_name == 'ALL':
        product_df = df.copy()
    else:
        product_df = df[df['Description'] == product_name].copy()
    
    if product_df.empty:
        return None
    
    # Group by frequency to get monthly/quarterly data
    product_df = product_df.set_index('Date').resample(freq)['qty'].sum().reset_index()
    product_df.columns = ['Date', 'qty']
    product_df.dropna(inplace=True)
    
    if len(product_df) < 2:
        return None
    
    total_units = product_df['qty'].sum()
    
    # Find highest and lowest sold periods
    highest_idx = product_df['qty'].idxmax()
    lowest_idx = product_df['qty'].idxmin()
    
    highest_sold = product_df.loc[highest_idx, 'qty']
    lowest_sold = product_df.loc[lowest_idx, 'qty']
    highest_month = product_df.loc[highest_idx, 'Date'].strftime('%Y-%m')
    lowest_month = product_df.loc[lowest_idx, 'Date'].strftime('%Y-%m')
    
    # Calculate trend for suggestion
    recent_sales = product_df['qty'].iloc[-min(3, len(product_df)):].mean()
    earlier_sales = product_df['qty'].iloc[:-min(3, len(product_df))].mean() if len(product_df) > 3 else product_df['qty'].iloc[0]
    trend_change = ((recent_sales - earlier_sales) / earlier_sales * 100) if earlier_sales > 0 else 0
    
    # Generate detailed suggestion with monthly insights
    if trend_change > 15:
        suggestion = f"Increase Stock - Best months: {highest_month}, Avoid: {lowest_month}"
    elif trend_change < -15:
        suggestion = f"Reduce Stock - Peak was: {highest_month}, Lowest: {lowest_month}"
    else:
        suggestion = f"Maintain Stock - Stable demand, Peak: {highest_month}"
    
    return {
        'product_name': product_name,
        'total_units': int(total_units),
        'highest_sold': int(highest_sold),
        'highest_month': highest_month,
        'lowest_sold': int(lowest_sold),
        'lowest_month': lowest_month,
        'suggestion': suggestion
    }

def generate_comprehensive_excel_report(df, location, category, freq):
    """Generate comprehensive Excel report for all products in the category"""
    
    # Create a BytesIO object to store the Excel file
    output = BytesIO()
    
    # Create Excel writer object
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        
        # Get all products in the category
        if category == 'ALL':
            products = sorted(df['Description'].unique())
        else:
            products = sorted(df[df['CategoryCode'] == category]['Description'].unique())
        
        # Create the main data for the report
        report_data = []
        
        # If "ALL" is selected for products, create one entry for all products combined
        if category == 'ALL':
            analysis = get_product_analysis(df, 'ALL', freq)
            if analysis:
                report_data.append({
                    'Product Name': 'ALL PRODUCTS COMBINED',
                    'Total Units Sold': analysis['total_units'],
                    'Highest Sold (Units)': analysis['highest_sold'],
                    'Best Performance Month': analysis['highest_month'],
                    'Lowest Sold (Units)': analysis['lowest_sold'],
                    'Lowest Performance Month': analysis['lowest_month'],
                    'Stock Suggestion': analysis['suggestion']
                })
        else:
            for product in products:
                analysis = get_product_analysis(df, product, freq)
                if analysis:
                    report_data.append({
                        'Product Name': analysis['product_name'],
                        'Total Units Sold': analysis['total_units'],
                        'Highest Sold (Units)': analysis['highest_sold'],
                        'Best Performance Month': analysis['highest_month'],
                        'Lowest Sold (Units)': analysis['lowest_sold'],
                        'Lowest Performance Month': analysis['lowest_month'],
                        'Stock Suggestion': analysis['suggestion']
                    })
        
        # Convert to DataFrame
        report_df = pd.DataFrame(report_data)
        
        # Write to Excel
        report_df.to_excel(writer, sheet_name='Sales Analysis', index=False)
        
        # Get the workbook and worksheet for formatting
        workbook = writer.book
        worksheet = writer.sheets['Sales Analysis']
        
        # Add formats
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#3498db',
            'font_color': 'white',
            'border': 1
        })
        
        cell_format = workbook.add_format({
            'text_wrap': True,
            'valign': 'top',
            'border': 1
        })
        
        # Apply header format
        for col_num, value in enumerate(report_df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Apply cell format and adjust column widths
        for col_num, column in enumerate(report_df.columns):
            max_length = max(
                report_df[column].astype(str).map(len).max(),
                len(str(column))
            )
            worksheet.set_column(col_num, col_num, min(max_length + 2, 50))
        
        # Add a summary sheet
        summary_data = {
            'Report Details': [
                'Location', 'Category', 'Total Products Analyzed', 
                'Report Generated', 'Analysis Frequency'
            ],
            'Values': [
                location if location != 'ALL' else 'All Locations',
                category if category != 'ALL' else 'All Categories',
                len(products) if category != 'ALL' else 1,
                pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
                freq
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Report Summary', index=False)
        
        # Format summary sheet
        summary_worksheet = writer.sheets['Report Summary']
        summary_worksheet.set_column(0, 0, 25)
        summary_worksheet.set_column(1, 1, 30)
    
    output.seek(0)
    return output

def generate_pdf_report(product_name, summary, fig):
    """Generate individual product PDF report"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Clean the product name
    clean_product_name = clean_text_for_pdf(product_name)
    pdf.cell(200, 10, txt=f"Sales Forecast Report - {clean_product_name}", ln=True, align='C')
    pdf.ln(10)
    
    # Clean all summary values
    for key, value in summary.items():
        clean_key = clean_text_for_pdf(str(key))
        clean_value = clean_text_for_pdf(str(value))
        # Split long lines
        if len(f"{clean_key}: {clean_value}") > 80:
            pdf.cell(200, 10, txt=f"{clean_key}:", ln=True)
            # Split value into multiple lines if too long
            words = clean_value.split()
            line = ""
            for word in words:
                if len(line + word) < 70:
                    line += word + " "
                else:
                    pdf.cell(200, 10, txt=f"  {line.strip()}", ln=True)
                    line = word + " "
            if line.strip():
                pdf.cell(200, 10, txt=f"  {line.strip()}", ln=True)
        else:
            pdf.cell(200, 10, txt=f"{clean_key}: {clean_value}", ln=True)
    
    # Add chart
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        fig.savefig(tmpfile.name, format="png", dpi=150, bbox_inches='tight')
        tmpfile_path = tmpfile.name
    
    # Adjust image position based on text length
    pdf.image(tmpfile_path, x=10, y=pdf.get_y() + 10, w=190)
    os.remove(tmpfile_path)
    
    # Generate PDF bytes
    try:
        pdf_output = pdf.output(dest='S')
        return BytesIO(pdf_output.encode('latin-1'))
    except UnicodeEncodeError:
        # Fallback: create a simpler PDF if encoding fails
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Sales Forecast Report", ln=True, align='C')
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Product: {clean_text_for_pdf(product_name)}", ln=True)
        pdf.cell(200, 10, txt=f"Report generated successfully", ln=True)
        pdf_output = pdf.output(dest='S')
        return BytesIO(pdf_output.encode('latin-1'))

def generate_excel_report_same_as_pdf(product_name, summary, actual_df, forecast_df):
    """Generate Excel report with same content as PDF"""
    
    # Create a BytesIO object to store the Excel file
    output = BytesIO()
    
    # Create Excel writer object
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        
        # Create summary sheet with same data as PDF
        summary_data = []
        for key, value in summary.items():
            summary_data.append({
                'Metric': key,
                'Value': str(value)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Report Summary', index=False)
        
        # Create forecast data sheet
        # Combine actual and forecast data
        excel_data = pd.merge(
            actual_df.rename(columns={'ds': 'Date', 'y': 'Actual_Sales'}),
            forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={
                'ds': 'Date', 'yhat': 'Forecasted_Sales', 
                'yhat_lower': 'Lower_Bound', 'yhat_upper': 'Upper_Bound'
            }),
            on='Date', how='outer'
        )
        
        excel_data.to_excel(writer, sheet_name='Forecast Data', index=False)
        
        # Get the workbook and worksheet for formatting
        workbook = writer.book
        
        # Format summary sheet
        summary_worksheet = writer.sheets['Report Summary']
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#3498db',
            'font_color': 'white',
            'border': 1
        })
        
        # Apply formatting
        summary_worksheet.set_column(0, 0, 25)
        summary_worksheet.set_column(1, 1, 50)
        
        # Format forecast data sheet
        forecast_worksheet = writer.sheets['Forecast Data']
        forecast_worksheet.set_column(0, 4, 15)
    
    output.seek(0)
    return output

# Load and preprocess data
@st.cache_data
def load_and_process_data():
    df = load_data()
    df = df[df['CategoryCode'].str.lower() != 'vegetables']
    return df

# Main app
st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide", initial_sidebar_state="expanded")

# Load custom CSS
load_css()

# Title
st.markdown('<h1 class="main-title">📊 Sales Forecasting Dashboard</h1>', unsafe_allow_html=True)

# Sidebar with custom styling and hamburger menu
if 'sidebar_state' not in st.session_state:
    st.session_state.sidebar_state = 'expanded'

with st.sidebar:
    
    st.markdown("### 🎛️ Control Panel")
    
    with st.spinner("Loading data..."):
        df = load_and_process_data()
    
    st.markdown("**📂 Category Selection**")
    category_options = ['ALL'] + sorted(df['CategoryCode'].dropna().unique())
    category = st.selectbox("Select Category", category_options, index=0, label_visibility="collapsed")
    
    st.markdown("**📍 Location Selection**") 
    location_options = ['ALL'] + sorted(df['LocationCode'].dropna().unique())
    location = st.selectbox("Select Location", location_options, label_visibility="collapsed")
    
    st.markdown("**📅 Frequency Selection**")
    freq_option = st.selectbox("Select Frequency", ["Monthly", "Quarterly", "Yearly"], label_visibility="collapsed")
    
    # Comprehensive report download button
    st.markdown("---")
    st.markdown("**📋 Comprehensive Report**")
    
    # Filter data for comprehensive report
    if location == 'ALL' and category == 'ALL':
        report_df = df.copy()
    elif location == 'ALL':
        report_df = df[df['CategoryCode'] == category]
    elif category == 'ALL':
        report_df = df[df['LocationCode'] == location]
    else:
        report_df = df[(df['CategoryCode'] == category) & (df['LocationCode'] == location)]
    
    freq_map = {"Monthly": "MS", "Quarterly": "QS", "Yearly": "YS"}
    freq = freq_map[freq_option]
    
    if st.button("📥 Download Category Report", type="primary"):
        with st.spinner("Generating comprehensive Excel report..."):
            excel_report = generate_comprehensive_excel_report(report_df, location, category, freq)
            
            # Create report title
            if location == 'ALL' and category == 'ALL':
                report_title = "ALL_LOCATIONS_ALL_CATEGORIES_Report.xlsx"
            elif location == 'ALL':
                report_title = f"ALL_LOCATIONS_{category}_Report.xlsx"
            elif category == 'ALL':
                report_title = f"{location}_ALL_CATEGORIES_Report.xlsx"
            else:
                report_title = f"{location}_{category}_Report.xlsx"
            
            st.download_button(
                label="📊 Download Excel Report",
                data=excel_report,
                file_name=report_title,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="comprehensive_report"
            )

# Filter data based on selections
freq_map = {"Monthly": "MS", "Quarterly": "QS", "Yearly": "YS"}
freq = freq_map[freq_option]

if location == 'ALL' and category == 'ALL':
    filtered_df = df.copy()
elif location == 'ALL':
    filtered_df = df[df['CategoryCode'] == category]
elif category == 'ALL':
    filtered_df = df[df['LocationCode'] == location]
else:
    filtered_df = df[(df['CategoryCode'] == category) & (df['LocationCode'] == location)]

# Get products for the selected category and location
if category == 'ALL':
    products = ['ALL'] + sorted(filtered_df['Description'].unique())
else:
    products = ['ALL'] + sorted(filtered_df['Description'].unique())

st.markdown("**🛍️ Product Selection**")
product_selected = st.selectbox("Select Product", products, index=0, label_visibility="collapsed")

if product_selected:
    # Calculate overall metrics
    if product_selected == 'ALL':
        total_units = filtered_df['qty'].sum()
        total_revenue = filtered_df['TaxableAmount'].sum()
        avg_monthly_sales = filtered_df.groupby(filtered_df['Date'].dt.to_period('M'))['qty'].sum().mean()
        unique_products = filtered_df['Description'].nunique()
    else:
        product_df = filtered_df[filtered_df['Description'] == product_selected]
        total_units = product_df['qty'].sum()
        total_revenue = product_df['TaxableAmount'].sum()
        avg_monthly_sales = product_df.groupby(product_df['Date'].dt.to_period('M'))['qty'].sum().mean()
        unique_products = 1

    # Display metrics in cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-title">📦 Total Units Sold</div>
            <div class="metric-value">{int(total_units):,}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-title">💰 Total Revenue</div>
            <div class="metric-value">₹{total_revenue:,.0f}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-title">📊 Avg Monthly Sales</div>
            <div class="metric-value">{int(avg_monthly_sales):,}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-title">🛍️ Products Analyzed</div>
            <div class="metric-value">{unique_products:,}</div>
        </div>
        ''', unsafe_allow_html=True)

    # Category trend analysis
    # st.markdown("### 📈 Category Performance Trends")
    
    # if category == 'ALL':
    #     categories_to_analyze = sorted(filtered_df['CategoryCode'].dropna().unique())
    # else:
    #     categories_to_analyze = [category]
    
    # trend_cols = st.columns(min(len(categories_to_analyze), 4))
    
    # for i, cat in enumerate(categories_to_analyze[:4]):  # Limit to 4 categories for display
    #     with trend_cols[i % 4]:
    #         trend_change, symbol, color = calculate_category_trend(filtered_df, cat, freq)
    #         st.markdown(f'''
    #         <div class="metric-card">
    #             <div class="metric-title">{cat}</div>
    #             <div class="metric-value" style="color: {color};">
    #                 {symbol} {trend_change:+.1f}%
    #             </div>
    #         </div>
    #         ''', unsafe_allow_html=True)

    # Forecasting section
    st.markdown("### 🔮 Sales Forecast & Analysis")
    
    with st.spinner("Generating forecast..."):
        forecast_result = forecast_sales(filtered_df, product_selected, freq)
    
    if forecast_result:
        model, df_forecast, actual_df = forecast_result
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot actual data
        actual_data = actual_df.copy()
        ax.plot(actual_data['ds'], actual_data['y'], 'o-', color='#2E86AB', linewidth=3, markersize=8, label='Actual Sales', alpha=0.8)
        
        # Plot forecast
        forecast_data = df_forecast[df_forecast['ds'] > actual_data['ds'].max()]
        if not forecast_data.empty:
            ax.plot(forecast_data['ds'], forecast_data['yhat'], 's-', color='#A23B72', linewidth=3, markersize=8, label='Forecasted Sales', alpha=0.8)
            
            # Add confidence intervals
            ax.fill_between(forecast_data['ds'], 
                          forecast_data['yhat_lower'], 
                          forecast_data['yhat_upper'], 
                          color='#A23B72', alpha=0.2, label='Confidence Interval')
        
        # Styling
        ax.set_title(f'Sales Forecast - {product_selected}', fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=14, fontweight='bold')
        ax.set_ylabel('Quantity Sold', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        
        # Display chart in container
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Analysis and recommendations
        trend_type, recommendation, icon = analyze_trend_and_recommend(actual_df, df_forecast)
        
        st.markdown("### 💡 Intelligent Stock Recommendations")
        
        recommendation_class = f"recommendation-{trend_type}"
        st.markdown(f'''
        <div class="recommendation-card {recommendation_class}">
            <div class="recommendation-title">{icon} Stock Management Insight</div>
            <div class="recommendation-text">{recommendation}</div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Summary statistics
        st.markdown("### 📋 Forecast Summary")
        
        summary_col1, summary_col2 = st.columns(2)
        
        with summary_col1:
            current_avg = actual_df['y'].tail(3).mean()
            forecast_avg = forecast_data['yhat'].mean() if not forecast_data.empty else 0
            
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-title">📊 Current Average ({freq_option})</div>
                <div class="metric-value">{current_avg:.0f} units</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with summary_col2:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-title">🔮 Forecast Average ({freq_option})</div>
                <div class="metric-value">{forecast_avg:.0f} units</div>
            </div>
            ''', unsafe_allow_html=True)
        
        # Individual product report downloads
        st.markdown("### 📄 Individual Product Reports")
        
        col1, col2 = st.columns(2)
        
        # Prepare summary for reports
        summary = {
            "Product": product_selected,
            "Analysis Period": f"{actual_df['ds'].min().strftime('%Y-%m-%d')} to {actual_df['ds'].max().strftime('%Y-%m-%d')}",
            "Total Units Sold": f"{actual_df['y'].sum():.0f}",
            "Average Sales": f"{actual_df['y'].mean():.2f}",
            "Trend": f"{trend_type.title()}",
            "Forecast Average": f"{forecast_avg:.0f}",
            "Recommendation": clean_text_for_pdf(recommendation.replace('*', '').replace('#', ''))
        }
        
        with col1:
            # PDF Report
            if st.button("📄 Generate PDF Report", key="pdf_btn"):
                with st.spinner("Generating PDF report..."):
                    pdf_buffer = generate_pdf_report(product_selected, summary, fig)
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_buffer,
                        file_name=f"{product_selected}_forecast_report.pdf",
                        mime="application/pdf"
                    )
        
        with col2:
            # Excel Report
            if st.button("📊 Generate Excel Report", key="excel_btn"):
                with st.spinner("Generating Excel report..."):
                    excel_buffer = generate_excel_report_same_as_pdf(product_selected, summary, actual_df, df_forecast)
                    st.download_button(
                        label="📥 Download Excel Report",
                        data=excel_buffer,
                        file_name=f"{product_selected}_forecast_report.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
    
    else:
        st.error("❌ Unable to generate forecast. Insufficient data for the selected product.")
        st.info("💡 Try selecting a different product or adjusting the time frequency.")

    # Data insights section
    st.markdown("### 🔍 Additional Insights")
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        # Top performing products
        if product_selected == 'ALL':
            top_products = filtered_df.groupby('Description')['qty'].sum().nlargest(5)
            st.markdown("**🏆 Top 5 Products by Sales**")
            for i, (product, sales) in enumerate(top_products.items(), 1):
                st.markdown(f"{i}. **{product}**: {sales:,} units")
    
    with insight_col2:
        # Monthly performance
        monthly_sales = filtered_df.groupby(filtered_df['Date'].dt.month)['qty'].sum()
        best_month = monthly_sales.idxmax()
        worst_month = monthly_sales.idxmin()
        
        st.markdown("**📅 Seasonal Insights**")
        st.markdown(f"🔥 **Best Month**: {pd.to_datetime(f'2023-{best_month}-01').strftime('%B')}")
        st.markdown(f"❄️ **Slowest Month**: {pd.to_datetime(f'2023-{worst_month}-01').strftime('%B')}")

else:
    st.info("👆 Please select a product to view the forecast analysis.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #7f8c8d; padding: 20px;'>
        <p>📊 Sales Forecasting Dashboard | Built with Streamlit</p>
        <p>🚀 Powered by Advanced Analytics & Machine Learning</p>
    </div>
    """, 
    unsafe_allow_html=True
)