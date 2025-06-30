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

def calculate_category_trend(df, category, freq):
    """Calculate category trend performance"""
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
        products = sorted(df['Description'].unique())
        
        # Create the main data for the report
        report_data = []
        
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
                category,
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

# Title
st.markdown('<h1 class="main-title">📊 Sales Forecasting Dashboard</h1>', unsafe_allow_html=True)

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
    category = st.selectbox("Select Category", sorted(df['CategoryCode'].dropna().unique()), label_visibility="collapsed")
    
    st.markdown("**📍 Location Selection**") 
    location_options = ['ALL'] + sorted(df['LocationCode'].dropna().unique())
    location = st.selectbox("Select Location", location_options, label_visibility="collapsed")
    
    st.markdown("**📅 Frequency Selection**")
    freq_option = st.selectbox("Select Frequency", ["Monthly", "Quarterly", "Yearly"], label_visibility="collapsed")
    
    # Comprehensive report download button
    st.markdown("---")
    st.markdown("**📋 Comprehensive Report**")
    
    # Filter data for comprehensive report
    if location == 'ALL':
        report_df = df[df['CategoryCode'] == category]
    else:
        report_df = df[(df['CategoryCode'] == category) & (df['LocationCode'] == location)]
    
    freq_map = {"Monthly": "MS", "Quarterly": "QS", "Yearly": "YS"}
    freq = freq_map[freq_option]
    
    if st.button("📥 Download Category Report", type="primary"):
        with st.spinner("Generating comprehensive Excel report..."):
            excel_report = generate_comprehensive_excel_report(report_df, location, category, freq)
            report_title = f"{location}_{category}_Report.xlsx" if location != 'ALL' else f"ALL_LOCATIONS_{category}_Report.xlsx"
            
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

if location == 'ALL':
    filtered_df = df[df['CategoryCode'] == category]
else:
    filtered_df = df[(df['CategoryCode'] == category) & (df['LocationCode'] == location)]

# Get all products for the selected category and location (no limit)
products = sorted(filtered_df['Description'].unique())

st.markdown("**🛍️ Product Selection**")
product_selected = st.selectbox("Select Product", products, label_visibility="collapsed")

if product_selected:
    # Calculate overall metrics
    product_data = filtered_df[filtered_df['Description'] == product_selected]
    total_qty = product_data['qty'].sum()
    total_amount = product_data['TaxableAmount'].sum()
    total_products = len(filtered_df['Description'].unique())
    
    # Calculate category trend
    trend_value, trend_symbol, trend_color = calculate_category_trend(filtered_df, category, freq)
    
    # Display metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">🛍️ Total Products</div>
            <div class="metric-value">{total_products:,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">📦 Total Quantity Sold</div>
            <div class="metric-value">{total_qty:,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">💰 Total Amount</div>
            <div class="metric-value">₹{total_amount:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">📈 Category Trend</div>
            <div class="metric-value" style="color: {trend_color};">
                <span style="font-size: 2rem;">{trend_symbol}</span> {trend_value:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Forecast analysis
    model_result = forecast_sales(filtered_df, product_selected, freq)
    if model_result:
        model, forecast, actual_df = model_result
        
        # Create and display chart
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Styling the plot
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('#f8f9fa')
        
        # Plot data
        ax.plot(actual_df['ds'], actual_df['y'], label='Actual Sales', marker='o', 
                linewidth=3, markersize=8, color='#2980b9')
        ax.plot(forecast['ds'], forecast['yhat'], label='Forecasted Sales', 
                linestyle='--', marker='x', linewidth=3, markersize=8, color='#e74c3c')
        ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                       alpha=0.3, color='#f39c12', label='Confidence Interval')

        # Add value annotations
        for i, row in actual_df.iterrows():
            ax.annotate(f"{row['y']:.0f}", (row['ds'], row['y']), 
                       textcoords="offset points", xytext=(0,10), ha='center', 
                       fontsize=10, fontweight='bold', color='#2c3e50')

        # Highlight max and min points
        if not actual_df.empty:
            max_point = actual_df.loc[actual_df['y'].idxmax()]
            min_point = actual_df.loc[actual_df['y'].idxmin()]
            ax.annotate("📈 Peak", (max_point['ds'], max_point['y']), 
                       textcoords="offset points", xytext=(0,20), ha='center', 
                       fontsize=11, fontweight='bold', color='#27ae60',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            ax.annotate("📉 Low", (min_point['ds'], min_point['y']), 
                       textcoords="offset points", xytext=(0,-30), ha='center', 
                       fontsize=11, fontweight='bold', color='#e74c3c',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

        # Styling
        ax.set_xlabel('Date', fontsize=14, fontweight='bold', color='#2c3e50')
        ax.set_ylabel('Quantity Sold', fontsize=14, fontweight='bold', color='#2c3e50')
        ax.set_title(f'Sales Forecast for {product_selected}', fontsize=16, fontweight='bold', 
                    color='#2c3e50', pad=20)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, 
                 bbox_to_anchor=(0.02, 0.98))
        
        # Format x-axis dates
        date_formatter = DateFormatter('%Y-%m')
        ax.xaxis.set_major_formatter(date_formatter)
        plt.xticks(rotation=45)
        
        # Improve layout
        plt.tight_layout()
        
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Analysis and recommendations
        trend_type, recommendation, icon = analyze_trend_and_recommend(actual_df, forecast)
        
        # Display recommendation with custom styling
        recommendation_class = f"recommendation-{trend_type}"
        st.markdown(f"""
        <div class="recommendation-card {recommendation_class}">
            <div class="recommendation-title">{icon} Stock Recommendation</div>
            <div class="recommendation-text">{recommendation}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Generate summary for PDF
        forecast_summary = {
            "Product": product_selected,
            "Category": category,
            "Location": location if location != 'ALL' else 'All Locations',
            "Analysis Period": f"{actual_df['ds'].min().strftime('%Y-%m')} to {actual_df['ds'].max().strftime('%Y-%m')}",
            "Total Units Sold": f"{total_qty:,}",
            # "Average Price per Unit": f"₹{avg_price:.2f}",
            "Peak Sales": f"{actual_df['y'].max():.0f} units in {actual_df.loc[actual_df['y'].idxmax(), 'ds'].strftime('%Y-%m')}",
            "Lowest Sales": f"{actual_df['y'].min():.0f} units in {actual_df.loc[actual_df['y'].idxmin(), 'ds'].strftime('%Y-%m')}",
            "Next Period Forecast": f"{forecast[forecast['ds'] > actual_df['ds'].max()]['yhat'].iloc[0]:.0f} units" if not forecast[forecast['ds'] > actual_df['ds'].max()].empty else "N/A",
            "Recommendation": clean_text_for_pdf(recommendation),
            "Analysis Frequency": freq_option,
            "Report Generated": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Download buttons
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            # PDF download
            if st.button("📄 Generate PDF Report", type="primary"):
                with st.spinner("Generating PDF report..."):
                    pdf_buffer = generate_pdf_report(product_selected, forecast_summary, fig)
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_buffer,
                        file_name=f"{product_selected}_forecast_report.pdf",
                        mime="application/pdf",
                        key="pdf_download"
                    )
        
        with col2:
            # Excel data download
            if st.button("📊 Export Data to Excel", type="secondary"):
                with st.spinner("Preparing Excel data..."):
                    # Combine actual and forecast data
                    excel_data = pd.merge(
                        actual_df.rename(columns={'ds': 'Date', 'y': 'Actual_Sales'}),
                        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={
                            'ds': 'Date', 'yhat': 'Forecasted_Sales', 
                            'yhat_lower': 'Lower_Bound', 'yhat_upper': 'Upper_Bound'
                        }),
                        on='Date', how='outer'
                    )
                    
                    # Create Excel buffer
                    excel_buffer = BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                        excel_data.to_excel(writer, sheet_name='Forecast Data', index=False)
                        
                        # Add summary sheet
                        summary_df = pd.DataFrame(list(forecast_summary.items()), columns=['Metric', 'Value'])
                        summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    
                    excel_buffer.seek(0)
                    
                    st.download_button(
                        label="📥 Download Excel Data",
                        data=excel_buffer,
                        file_name=f"{product_selected}_forecast_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="excel_download"
                    )
        
        # Additional insights
        st.markdown("---")
        st.markdown("### 📈 Additional Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">📊 Sales Trend</div>
                <div class="metric-value" style="font-size: 1.5rem;">
                    {"📈 Growing" if trend_type == "positive" else "📉 Declining" if trend_type == "negative" else "📊 Stable"}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            volatility = actual_df['y'].std() / actual_df['y'].mean() * 100 if actual_df['y'].mean() > 0 else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">📉 Sales Volatility</div>
                <div class="metric-value" style="font-size: 1.5rem;">
                    {volatility:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            growth_rate = ((actual_df['y'].iloc[-1] - actual_df['y'].iloc[0]) / actual_df['y'].iloc[0] * 100) if len(actual_df) > 1 and actual_df['y'].iloc[0] > 0 else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">📈 Overall Growth</div>
                <div class="metric-value" style="font-size: 1.5rem; color: {'#27ae60' if growth_rate > 0 else '#e74c3c'};">
                    {growth_rate:+.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Seasonal analysis (if monthly data)
        if freq_option == "Monthly" and len(actual_df) >= 12:
            st.markdown("### 🗓️ Seasonal Analysis")
            actual_df['Month'] = actual_df['ds'].dt.month
            monthly_avg = actual_df.groupby('Month')['y'].mean().sort_values(ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🏆 Top Performing Months:**")
                for i, (month, avg_sales) in enumerate(monthly_avg.head(3).items()):
                    month_name = pd.Timestamp(2024, month, 1).strftime('%B')
                    st.markdown(f"**{i+1}.** {month_name}: {avg_sales:.0f} units")
            
            with col2:
                st.markdown("**📉 Lowest Performing Months:**")
                for i, (month, avg_sales) in enumerate(monthly_avg.tail(3).items()):
                    month_name = pd.Timestamp(2024, month, 1).strftime('%B')
                    st.markdown(f"**{i+1}.** {month_name}: {avg_sales:.0f} units")
        
    else:
        st.error("❌ Unable to generate forecast. Insufficient data for the selected product.")
        st.info("💡 Try selecting a different product or adjusting the frequency settings.")

else:
    st.info("👆 Please select a product from the sidebar to view its sales forecast.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d; font-size: 0.9rem; margin-top: 2rem;">
    <p>📊 <strong>Sales Forecasting Dashboard</strong> | Built with Streamlit & Machine Learning</p>
    <p>🔬 Powered by Linear Regression | 📈 Real-time Analytics</p>
</div>
""", unsafe_allow_html=True)