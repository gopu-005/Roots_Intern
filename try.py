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
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin: 2rem 0;
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
    
    /* Product summary card */
    .product-summary-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #3498db;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    
    .product-name {
        font-size: 1.1rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    
    .product-metrics {
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 1rem;
    }
    
    .product-metric {
        text-align: center;
        flex: 1;
        min-width: 120px;
    }
    
    .product-metric-value {
        font-size: 1.2rem;
        font-weight: bold;
        color: #3498db;
    }
    
    .product-metric-label {
        font-size: 0.9rem;
        color: #7f8c8d;
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

def forecast_overall_sales(df, freq):
    """Generate forecast for overall sales data"""
    # Aggregate all sales by date
    df_agg = df.set_index('Date').resample(freq)['qty'].sum().reset_index()
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
    future_dates = pd.date_range(start=last_date, periods=6, freq=freq)[1:]
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
    avg_units = product_df['qty'].mean()
    
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
        suggestion = f"📈 Increase Stock - Best months: {highest_month}, Avoid: {lowest_month}"
        trend_status = "Growing"
    elif trend_change < -15:
        suggestion = f"📉 Reduce Stock - Peak was: {highest_month}, Lowest: {lowest_month}"
        trend_status = "Declining"
    else:
        suggestion = f"📊 Maintain Stock - Stable demand, Peak: {highest_month}"
        trend_status = "Stable"
    
    return {
        'product_name': product_name,
        'total_units': int(total_units),
        'avg_units': int(avg_units),
        'highest_sold': int(highest_sold),
        'highest_month': highest_month,
        'lowest_sold': int(lowest_sold),
        'lowest_month': lowest_month,
        'trend_change': trend_change,
        'trend_status': trend_status,
        'suggestion': suggestion
    }

def generate_all_products_summary(df, category, location, freq):
    """Generate summary for all products"""
    # Filter data
    if category == 'ALL':
        filtered_df = df.copy()
    else:
        filtered_df = df[df['CategoryCode'] == category].copy()
    
    if location != 'ALL':
        filtered_df = filtered_df[filtered_df['LocationCode'] == location]
    
    # Get all products
    products = sorted(filtered_df['Description'].unique())
    
    # Analyze each product
    product_summaries = []
    for product in products:
        analysis = get_product_analysis(filtered_df, product, freq)
        if analysis:
            product_summaries.append(analysis)
    
    return product_summaries

def generate_comprehensive_excel_report(df, category, location, freq):
    """Generate comprehensive Excel report for all products in the category"""
    
    # Create a BytesIO object to store the Excel file
    output = BytesIO()
    
    # Create Excel writer object
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        
        # Filter data
        if category == 'ALL':
            filtered_df = df.copy()
        else:
            filtered_df = df[df['CategoryCode'] == category].copy()
        
        if location != 'ALL':
            filtered_df = filtered_df[filtered_df['LocationCode'] == location]
        
        # Get all products
        products = sorted(filtered_df['Description'].unique())
        
        # Create the main data for the report
        report_data = []
        
        for product in products:
            analysis = get_product_analysis(filtered_df, product, freq)
            if analysis:
                report_data.append({
                    'Product Name': analysis['product_name'],
                    'Total Units Sold': analysis['total_units'],
                    'Average Units': analysis['avg_units'],
                    'Highest Sold (Units)': analysis['highest_sold'],
                    'Best Performance Month': analysis['highest_month'],
                    'Lowest Sold (Units)': analysis['lowest_sold'],
                    'Lowest Performance Month': analysis['lowest_month'],
                    'Trend Change (%)': f"{analysis['trend_change']:.1f}%",
                    'Trend Status': analysis['trend_status'],
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
                len(products),
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

# Load and preprocess data
@st.cache_data
def load_and_process_data():
    df = load_data()
    df = df[df['CategoryCode'].str.lower() != 'vegetables']
    # Removed the top 75 filter - now returns all products
    return df

# Main app
st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide", initial_sidebar_state="expanded")

# Load custom CSS
load_css()

# Title 📊 
st.markdown('<h1 class="main-title"> Veyr Organics Forecasting Dashboard</h1>', unsafe_allow_html=True)

# Sidebar with custom styling and hamburger menu
if 'sidebar_state' not in st.session_state:
    st.session_state.sidebar_state = 'expanded'

# Sidebar toggle button
st.markdown("""
<script>
function toggleSidebar() {
    const sidebar = document.querySelector('[data-testid="stSidebar"]');
    if (sidebar.style.marginLeft === '-21rem') {
        sidebar.style.marginLeft = '0';
    } else {
        sidebar.style.marginLeft = '-21rem';
    }
}
</script>
""", unsafe_allow_html=True)

with st.sidebar:
    
    st.markdown("### 🎛️ Control Panel")
    
    with st.spinner("Loading data..."):
        df = load_and_process_data()
    
    st.markdown("**📂 Category Selection**")
    category_options = ['ALL'] + sorted(df['CategoryCode'].dropna().unique())
    category = st.selectbox("Select Category", category_options, label_visibility="collapsed")
    
    st.markdown("**📍 Location Selection**") 
    location_options = ['ALL'] + sorted(df['LocationCode'].dropna().unique())
    location = st.selectbox("Select Location", location_options, label_visibility="collapsed")
    
    st.markdown("**📅 Frequency Selection**")
    freq_option = st.selectbox("Select Frequency", ["Monthly", "Quarterly", "Yearly"], label_visibility="collapsed")
    
    # Comprehensive report download button
    st.markdown("---")
    st.markdown("**📋 Comprehensive Report**")
    
    # Filter data for comprehensive report
    if category == 'ALL':
        report_df = df.copy()
    else:
        report_df = df[df['CategoryCode'] == category]
    
    if location != 'ALL':
        report_df = report_df[report_df['LocationCode'] == location]
    
    freq_map = {"Monthly": "MS", "Quarterly": "QS", "Yearly": "YS"}
    freq = freq_map[freq_option]
    
    if st.button("📥 Download Category Report", type="primary"):
        with st.spinner("Generating comprehensive Excel report..."):
            excel_report = generate_comprehensive_excel_report(report_df, category, location, freq)
            if category == 'ALL':
                report_title = f"ALL_CATEGORIES_{location}_Report.xlsx" if location != 'ALL' else "ALL_CATEGORIES_ALL_LOCATIONS_Report.xlsx"
            else:
                report_title = f"{location}_{category}_Report.xlsx" if location != 'ALL' else f"ALL_LOCATIONS_{category}_Report.xlsx"
            
            st.download_button(
                label="📊 Download Excel Report",
                data=excel_report,
                file_name=report_title,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    # Add some info about the report
    st.markdown("---")
    st.markdown("### 📊 Report Information")
    st.markdown(f"**Selected Category:** {category}")
    st.markdown(f"**Selected Location:** {location}")
    st.markdown(f"**Analysis Frequency:** {freq_option}")
    total_products = len(report_df['Description'].unique())
    st.markdown(f"**Total Products:** {total_products}")
    
    # Add trend analysis for the selected category
    trend_change, trend_symbol, trend_color = calculate_category_trend(report_df, category, freq)
    st.markdown(f"**Category Trend:** <span style='color:{trend_color}; font-size:1.2em;'>{trend_symbol} {trend_change:.1f}%</span>", unsafe_allow_html=True)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📈 Sales Analysis")
    
    # Filter data based on sidebar selections
    if category == 'ALL':
        filtered_df = df.copy()
    else:
        filtered_df = df[df['CategoryCode'] == category]
    
    if location != 'ALL':
        filtered_df = filtered_df[filtered_df['LocationCode'] == location]
    
    # Overall sales forecast
    st.markdown("#### 🎯 Overall Sales Forecast")
    
    freq_map = {"Monthly": "MS", "Quarterly": "QS", "Yearly": "YS"}
    freq = freq_map[freq_option]
    
    # Generate overall forecast
    overall_forecast = forecast_overall_sales(filtered_df, freq)
    
    if overall_forecast:
        model, forecast_df, historical_df = overall_forecast
        
        # Plot overall sales
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot historical data
        ax.plot(historical_df['ds'], historical_df['y'], 'o-', label='Historical Sales', color='#3498db', linewidth=2, markersize=6)
        
        # Plot forecast
        future_df = forecast_df[forecast_df['ds'] > historical_df['ds'].max()]
        ax.plot(future_df['ds'], future_df['yhat'], 'o-', label='Forecast', color='#e74c3c', linewidth=2, markersize=6)
        
        # Fill confidence interval
        ax.fill_between(future_df['ds'], future_df['yhat_lower'], future_df['yhat_upper'], 
                       alpha=0.3, color='#e74c3c', label='Confidence Interval')
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Sales Quantity', fontsize=12)
        ax.set_title(f'Overall Sales Forecast - {category} ({freq_option})', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format dates on x-axis
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        
        st.pyplot(fig)
        plt.close()
        
        # Overall analysis and recommendations
        overall_type, overall_recommendation, overall_emoji = analyze_trend_and_recommend(historical_df, forecast_df)
        
        # Display overall recommendation
        if overall_type == "positive":
            st.markdown(f"""
            <div class="recommendation-card recommendation-positive">
                <div class="recommendation-title">{overall_emoji} Overall Category Recommendation</div>
                <div class="recommendation-text">{overall_recommendation}</div>
            </div>
            """, unsafe_allow_html=True)
        elif overall_type == "negative":
            st.markdown(f"""
            <div class="recommendation-card recommendation-negative">
                <div class="recommendation-title">{overall_emoji} Overall Category Recommendation</div>
                <div class="recommendation-text">{overall_recommendation}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="recommendation-card recommendation-neutral">
                <div class="recommendation-title">{overall_emoji} Overall Category Recommendation</div>
                <div class="recommendation-text">{overall_recommendation}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Product-specific analysis
    st.markdown("#### 🔍 Product-Specific Analysis")
    
    # Get products for the selected category and location
    if category == 'ALL':
        products = sorted(filtered_df['Description'].dropna().unique())
    else:
        products = sorted(filtered_df[filtered_df['CategoryCode'] == category]['Description'].dropna().unique())
    
    if products:
        selected_product = st.selectbox("Select Product for Detailed Analysis", products)
        
        # Generate forecast for selected product
        product_forecast = forecast_sales(filtered_df, selected_product, freq)
        
        if product_forecast:
            model, forecast_df, historical_df = product_forecast
            
            # Plot product-specific forecast
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot historical data
            ax.plot(historical_df['ds'], historical_df['y'], 'o-', label='Historical Sales', color='#3498db', linewidth=2, markersize=6)
            
            # Plot forecast
            future_df = forecast_df[forecast_df['ds'] > historical_df['ds'].max()]
            ax.plot(future_df['ds'], future_df['yhat'], 'o-', label='Forecast', color='#e74c3c', linewidth=2, markersize=6)
            
            # Fill confidence interval
            ax.fill_between(future_df['ds'], future_df['yhat_lower'], future_df['yhat_upper'], 
                           alpha=0.3, color='#e74c3c', label='Confidence Interval')
            
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Sales Quantity', fontsize=12)
            ax.set_title(f'Sales Forecast: {selected_product} ({freq_option})', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Format dates on x-axis
            ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
            plt.xticks(rotation=45)
            
            st.pyplot(fig)
            plt.close()
            
            # Product analysis and recommendations
            product_type, product_recommendation, product_emoji = analyze_trend_and_recommend(historical_df, forecast_df)
            
            # Display product recommendation
            if product_type == "positive":
                st.markdown(f"""
                <div class="recommendation-card recommendation-positive">
                    <div class="recommendation-title">{product_emoji} Product Recommendation</div>
                    <div class="recommendation-text">{product_recommendation}</div>
                </div>
                """, unsafe_allow_html=True)
            elif product_type == "negative":
                st.markdown(f"""
                <div class="recommendation-card recommendation-negative">
                    <div class="recommendation-title">{product_emoji} Product Recommendation</div>
                    <div class="recommendation-text">{product_recommendation}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="recommendation-card recommendation-neutral">
                    <div class="recommendation-title">{product_emoji} Product Recommendation</div>
                    <div class="recommendation-text">{product_recommendation}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Individual product report download
            st.markdown("---")
            col_pdf, col_space = st.columns([1, 2])
            
            with col_pdf:
                # Generate summary for PDF
                analysis = get_product_analysis(filtered_df, selected_product, freq)
                if analysis:
                    summary = {
                        'Product': selected_product,
                        'Total Units Sold': f"{analysis['total_units']:,}",
                        'Average Units per Period': f"{analysis['avg_units']:,}",
                        'Highest Sales': f"{analysis['highest_sold']:,} units in {analysis['highest_month']}",
                        'Lowest Sales': f"{analysis['lowest_sold']:,} units in {analysis['lowest_month']}",
                        'Trend': f"{analysis['trend_change']:.1f}% ({analysis['trend_status']})",
                        'Recommendation': clean_text_for_pdf(analysis['suggestion'])
                    }
                    
                    pdf_report = generate_pdf_report(selected_product, summary, fig)
                    
                    st.download_button(
                        label="📄 Download Product Report (PDF)",
                        data=pdf_report,
                        file_name=f"{selected_product}_forecast_report.pdf",
                        mime="application/pdf"
                    )
        else:
            st.warning("⚠️ Insufficient data for forecasting this product. Please select a different product.")
    else:
        st.warning("⚠️ No products found for the selected category and location.")

with col2:
    st.markdown("### 📊 Quick Stats")
    
    # Calculate key metrics
    total_sales = filtered_df['qty'].sum()
    avg_sales = filtered_df['qty'].mean()
    unique_products = filtered_df['Description'].nunique()
    date_range = f"{filtered_df['Date'].min().strftime('%Y-%m')} to {filtered_df['Date'].max().strftime('%Y-%m')}"
    
    # Display metrics in cards
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Total Sales</div>
        <div class="metric-value">{total_sales:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Average Sales</div>
        <div class="metric-value">{avg_sales:.1f}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Unique Products</div>
        <div class="metric-value">{unique_products}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Date Range</div>
        <div class="metric-value" style="font-size: 1.2rem;">{date_range}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Product summary section
    st.markdown("### 📋 Product Summary")
    
    # Generate product summaries
    product_summaries = generate_all_products_summary(filtered_df, category, location, freq)
    
    # Display top 10 products summary
    if product_summaries:
        # Sort by total units sold
        top_products = sorted(product_summaries, key=lambda x: x['total_units'], reverse=True)[:10]
        
        st.markdown("#### 🏆 Top 10 Products")
        
        for i, product in enumerate(top_products, 1):
            # Determine color based on trend
            if product['trend_change'] > 15:
                border_color = "#27ae60"
                trend_color = "#27ae60"
            elif product['trend_change'] < -15:
                border_color = "#e74c3c"
                trend_color = "#e74c3c"
            else:
                border_color = "#f39c12"
                trend_color = "#f39c12"
            
            st.markdown(f"""
            <div class="product-summary-card" style="border-left-color: {border_color};">
                <div class="product-name">#{i}. {product['product_name'][:30]}{'...' if len(product['product_name']) > 30 else ''}</div>
                <div class="product-metrics">
                    <div class="product-metric">
                        <div class="product-metric-value">{product['total_units']:,}</div>
                        <div class="product-metric-label">Total Units</div>
                    </div>
                    <div class="product-metric">
                        <div class="product-metric-value">{product['avg_units']:,}</div>
                        <div class="product-metric-label">Avg Units</div>
                    </div>
                    <div class="product-metric">
                        <div class="product-metric-value" style="color: {trend_color};">{product['trend_change']:.1f}%</div>
                        <div class="product-metric-label">Trend</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No product data available for analysis.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d; margin-top: 2rem;">
    <p>📊 Sales Forecasting Dashboard | Built with Streamlit</p>
    <p>📧 For support or questions, contact your analytics team</p>
</div>
""", unsafe_allow_html=True)