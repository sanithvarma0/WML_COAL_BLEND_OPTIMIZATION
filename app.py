import streamlit as st
import pandas as pd
import numpy as np
import io
import sys
from contextlib import redirect_stdout, redirect_stderr
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from coal_blend_optimizer import optimize_coal_blend_enhanced

# Custom CSS for enhanced styling
def load_custom_css():
    st.markdown("""
    <style>
    /* Main app styling */
    .main > div {
        padding-top: 2rem;
    }
    
    /* Custom color scheme */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --success-color: #2ca02c;
        --warning-color: #ff7f0e;
        --danger-color: #d62728;
        --dark-bg: #0e1117;
        --card-bg: #262730;
    }
    
    /* Header styling */               
    .main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem 1rem;  /* Changed from 2rem 1rem */
    border-radius: 10px;  /* Changed from 15px */
    color: white;
    text-align: center;
    margin-bottom: 1.5rem;  /* Changed from 2rem */
    box-shadow: 0 8px 20px rgba(0,0,0,0.2);  /* Reduced shadow */
    }
    
    .main-header h3 {  /* Changed from h1 to h3 */
    font-size: 2rem;
    font-weight: 700;
    margin: 0;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
    font-size: 1rem;  /* Changed from 1.2rem */
    margin: 0.3rem 0 0 0;  /* Changed from 0.5rem */
    opacity: 0.9;
    }
    
    /* Card styling */
    .stCard {
        background: var(--card-bg);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.1);
        margin: 1rem 0;
    }
    
    /* Custom metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
        margin: 0.5rem 0 0 0;
    }
    
    /* Status indicators */
    .status-success {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 600;
        text-align: center;
        display: inline-block;
    }
    
    .status-warning {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 600;
        text-align: center;
        display: inline-block;
    }
    
    .status-error {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 600;
        text-align: center;
        display: inline-block;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    
    /* Form styling */
    .stForm {
        background: rgba(255,255,255,0.05);
        border-radius: 15px;
        padding: 2rem;
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;  /* Increased from 8px for more spacing */
    }

    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        color: white;
        font-weight: 600;
        padding: 10px 22px;  /* Add this line - increases internal padding */
        min-width: 100px;    /* Add this line - sets minimum width */
        text-align: center;  /* Add this line - centers the text */
    }

    .stTabs [data-baseweb="tab"] span {  /* Add this new rule */
        font-size: 16px;     /* Increases font size */
        white-space: nowrap; /* Prevents text wrapping */
    }
    
    /* Input styling */
    .stNumberInput > div > div > input {
        border-radius: 10px;
        border: 2px solid rgba(255,255,255,0.2);
        background: rgba(255,255,255,0.1);
        transition: all 0.3s ease;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Animation keyframes */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .animate-fade-in {
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* Custom spacing */
    .section-spacing {
        margin: 3rem 0;
    }
    
    /* Tooltip styling */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: pointer;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: white;
        text-align: center;
        border-radius: 6px;
        padding: 5px 10px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    </style>
    """, unsafe_allow_html=True)

def create_header():
    st.markdown("""
    <div class="main-header animate-fade-in">
        <h3>🏭 Coal Blend Optimizer</h3>
        <p>Advanced Multi-Parameter Optimization System</p>
    </div>
    """, unsafe_allow_html=True)

def create_metric_card(title, value, color="primary"):
    color_map = {
        "primary": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        "success": "linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%)",
        "warning": "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
        "danger": "linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%)"
    }
    
    st.markdown(f"""
    <div class="metric-card" style="background: {color_map[color]};">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{title}</div>
    </div>
    """, unsafe_allow_html=True)

def create_status_badge(status, text):
    status_class = f"status-{status}"
    st.markdown(f'<div class="{status_class}">{text}</div>', unsafe_allow_html=True)

def create_silo_visualization(silo_properties):
    """Create a simple bar chart for silo properties comparison"""
    
    # Prepare data
    data = []
    properties = ['Ash%', 'I.M.%', 'V.M.%', 'GM%', 'F.C.%', 'CSN']
    
    for i, silo_dict in enumerate(silo_properties, 1):
        if i <= 5:
            for prop in properties:
                key = f'SILO_{i}_{prop}'
                value = silo_dict.get(key, 0)
                data.append({
                    'Silo': f'SILO {i}',
                    'Property': prop,
                    'Value': value
                })
    
    df = pd.DataFrame(data)
    
    # Create grouped bar chart
    fig = px.bar(df, x='Property', y='Value', color='Silo',
                 title="Silo Properties Comparison",
                 barmode='group')
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

def create_discharge_chart(optimal_discharges, active_silos):
    """Create a beautiful discharge visualization"""
    
    silos = [f'SILO {i}' for i in range(1, len(optimal_discharges) + 1)]
    colors = ['#2E8B57' if i+1 in active_silos else '#D3D3D3' for i in range(len(optimal_discharges))]
    
    fig = go.Figure(data=[
        go.Bar(
            x=silos,
            y=optimal_discharges,
            marker=dict(
                color=colors,
                line=dict(color='rgba(0,0,0,0.8)', width=2),
                opacity=0.8
            ),
            text=[f'{val:.1f}%' for val in optimal_discharges],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='Optimal Discharge Distribution',
        xaxis_title='Silos',
        yaxis_title='Discharge (%)',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=False,
        height=400
    )
    
    return fig

def create_properties_gauge_chart(predictions, target_ranges):
    """Create gauge charts for blend properties"""
    
    properties = list(predictions.keys())
    
    # Create subplot with gauge charts
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[prop.replace('BLEND_', '') for prop in properties],
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}],
               [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]]
    )
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    for i, (prop, value) in enumerate(predictions.items()):
        row = i // 3 + 1
        col = i % 3 + 1
        
        target_min, target_max = target_ranges[prop]
        
        # Determine color based on whether value is in range
        if target_min <= value <= target_max:
            color = "#2E8B57"  # Green for success
        else:
            color = "#DC143C"  # Red for violation
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=value,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': prop.replace('BLEND_', '')},
                gauge={
                    'axis': {'range': [None, target_max * 1.2]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, target_min], 'color': "lightgray"},
                        {'range': [target_min, target_max], 'color': "lightgreen"},
                        {'range': [target_max, target_max * 1.2], 'color': "lightgray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': target_max
                    }
                }
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        height=600,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "white", 'family': "Arial"}
    )
    
    return fig

def capture_output(func, *args, **kwargs):
    """Enhanced function to capture all output including errors"""
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            result = func(*args, **kwargs)
        
        output = stdout_capture.getvalue()
        error_output = stderr_capture.getvalue()
        
        # Combine outputs
        combined_output = output
        if error_output:
            combined_output += "\n--- ERRORS ---\n" + error_output
            
        return result, combined_output
    except Exception as e:
        error_msg = f"Exception occurred: {str(e)}\n{stdout_capture.getvalue()}\n{stderr_capture.getvalue()}"
        return None, error_msg

def format_blend_properties(predictions, target_ranges):
    """Format blend properties for display"""
    formatted = []
    for prop, value in predictions.items():
        target_min, target_max = target_ranges[prop]
        center = (target_min + target_max) / 2
        
        if target_min <= value <= target_max:
            status = "SUCCESS"
            deviation_from_center = abs(value - center)
            indicator = f"(center deviation: {deviation_from_center:.2f})"
        else:
            status = "VIOLATION"
            if value < target_min:
                indicator = f"(below by {target_min - value:.2f})"
            else:
                indicator = f"(above by {value - target_max:.2f})"
        
        formatted.append({
            'Property': prop.replace('BLEND_', ''),
            'Value': f"{value:.2f}",
            'Target Range': f"{target_min:.1f} - {target_max:.1f}",
            'Status': status,
            'Indicator': indicator
        })
    return formatted

def format_silo_properties(silo_properties):
    data = []

    for silo_dict in silo_properties:   # loop through list of silo dicts
        for key, value in silo_dict.items():
            parts = key.split("_")  # e.g. ["SILO", "1", "Ash%"]
            silo_no = "_".join(parts[:2])  # -> "SILO_1"
            prop = parts[2]                # -> "Ash%"
            # Find row for this silo or create new
            row = next((r for r in data if r["SILO_NO"] == silo_no), None)
            if not row:
                row = {"SILO_NO": silo_no}
                data.append(row)
            # Map property names cleanly
            if prop == "Ash%":
                row["Ash"] = value
            elif prop == "I.M.%":
                row["I.M."] = value
            elif prop == "V.M.%":
                row["V.M."] = value
            elif prop == "GM%":
                row["G.M."] = value
            elif prop == "F.C.%":
                row["F.C."] = value
            elif prop == "CSN":
                row["CSN"] = value
    # Build dataframe
    df = pd.DataFrame(data, columns=["SILO_NO", "Ash", "I.M.", "V.M.", "G.M.", "F.C.", "CSN"])
    return df.sort_values("SILO_NO").reset_index(drop=True)

def main():
    st.set_page_config(
        page_title="Coal Blend Optimizer Pro",
        page_icon="🏭",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state for example data flag
    if 'use_example_data' not in st.session_state:
        st.session_state['use_example_data'] = False
    
    # Load custom CSS
    load_custom_css()
    
    # Create header
    create_header()
    
    # Enhanced Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Optimization Parameters")
        
        with st.container():
            n_historical_starts = st.number_input(
                "🔄 Historical Starting Points", 
                min_value=1, max_value=10, value=3,
                key="hist_starts",
                help="Number of historical data points to use as starting points"
            )
            
            n_random_starts = st.number_input(
                "🎲 Random Starting Points", 
                min_value=1, max_value=10, value=3,
                key="rand_starts",
                help="Number of random starting points for optimization"
            )
            print(f"Historical Starts: {n_historical_starts}, Random Starts: {n_random_starts}")
        
        st.markdown("---")
        st.markdown("### 🏗️ Active Silos")
        
        # Enhanced silo selection with visual indicators
        silo_options = [1, 2, 3, 4, 5]
        active_silos = st.multiselect(
            "Select Active Silos",
            silo_options,
            default=[1, 2, 3, 4, 5],
            help="Choose which silos are available for optimization"
        )
        
        # Visual indicator for active silos
        if active_silos:
            st.success(f"✅ {len(active_silos)} Silos Active")
            # for silo in active_silos:
            #     st.markdown(f"• SILO {silo}")
        else:
            st.error("❌ No Active Silos")
        
        st.markdown("---")
        st.markdown("### 📊 Target Quality Parameters")
        
        target_info = {
            "V.M.%": "24.0 - 25.0",
            "CSN": "4.5 - 6.0",
            "Ash%": "0.0 - 10.0",
            "F.C.%": "60.0 - 68.0",
            "GM%": "9.0 - 11.0",
            "I.M.%": "0.0 - 1.0"
        }
        
        for param, range_val in target_info.items():
            st.markdown(f"**{param}:** {range_val}")
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Enhanced load example data section
        st.markdown("### 📥 Data Input")
        
        # FIXED: Move button outside form and update session state properly
        if st.button("🎯 Load Example Data", help="Load sample coal property data"):
            st.session_state['use_example_data'] = True
            st.rerun()
        
        if st.session_state.get('use_example_data', False):
            st.info("📋 Using example coal property values")
        
        # Enhanced input form
        with st.form("silo_properties_form"):
            st.markdown("### 🧪 Coal Properties Configuration")
            
            # Default values
            if st.session_state.get('use_example_data', False):
                default_values = [
                        # Silo 1
                        {'ash': 9.12, 'im': 0.62, 'vm': 23.32, 'gm': 6.41, 'fc': 66.94, 'csn': 8.0},
                        # Silo 2
                        {'ash': 7.66, 'im': 0.48, 'vm': 24.22, 'gm': 9.33, 'fc': 67.64, 'csn': 8.0},
                        # Silo 3
                        {'ash': 8.83, 'im': 0.92, 'vm': 32.72, 'gm': 11.76, 'fc': 57.53, 'csn': 7.5},
                        # Silo 4
                        {'ash': 10.15, 'im': 0.66, 'vm': 23.88, 'gm': 9.59, 'fc': 65.31, 'csn': 8.5},
                        # Silo 5
                        {'ash': 9.45, 'im': 0.72, 'vm': 20.20, 'gm': 10.96, 'fc': 69.63, 'csn': 1.0}
                    ]

            else:
                default_values = [
                    {'ash': 10.0, 'im': 0.8, 'vm': 25.0, 'gm': 8.0, 'fc': 65.0, 'csn': 5.0}
                ] * 5
            
            # Enhanced tabs with icons
            tab_names = [f"🏗️ SILO {i}" for i in range(1, 6)]
            silo_tabs = st.tabs(tab_names)
            
            silo_properties = []
            
            for i, tab in enumerate(silo_tabs, 1):
                defaults = default_values[i-1] if i <= len(default_values) else default_values[0]
                
                with tab:
                    # Status indicator for each silo
                    if i in active_silos:
                        st.success(f"✅ SILO {i} - ACTIVE")
                    else:
                        st.warning(f"⚠️ SILO {i} - INACTIVE")
                    
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        ash = st.number_input(f"🔥 Ash% (SILO_{i})", min_value=0.0, max_value=50.0, value=defaults['ash'], step=0.1, key=f"silo_{i}_ash")
                        im = st.number_input(f"💧 I.M.% (SILO_{i})", min_value=0.0, max_value=10.0, value=defaults['im'], step=0.01, key=f"silo_{i}_im")
                    
                    with col_b:
                        vm = st.number_input(f"💨 V.M.% (SILO_{i})", min_value=0.0, max_value=50.0, value=defaults['vm'], step=0.1, key=f"silo_{i}_vm")
                        gm = st.number_input(f"⚗️ GM% (SILO_{i})", min_value=0.0, max_value=20.0, value=defaults['gm'], step=0.1, key=f"silo_{i}_gm")
                    
                    with col_c:
                        fc = st.number_input(f"🔗 F.C.% (SILO_{i})", min_value=0.0, max_value=100.0, value=defaults['fc'], step=0.1, key=f"silo_{i}_fc")
                        csn = st.number_input(f"🏭 CSN (SILO_{i})", min_value=0.0, max_value=10.0, value=defaults['csn'], step=0.1, key=f"silo_{i}_csn")
                    
                    # Create silo property dictionary
                    silo_props = {
                        f'SILO_{i}_Ash%': ash,
                        f'SILO_{i}_I.M.%': im,
                        f'SILO_{i}_V.M.%': vm,
                        f'SILO_{i}_GM%': gm,
                        f'SILO_{i}_F.C.%': fc,
                        f'SILO_{i}_CSN': csn
                    }
                    silo_properties.append(silo_props)
            
            # Enhanced submit button
            submitted = st.form_submit_button("🚀 Optimize Coal Blend", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("### 📊 System Overview")
        
        # Create overview cards
        if active_silos:
            create_metric_card("Active Silos", len(active_silos), "success")
        else:
            create_metric_card("Active Silos", "0", "danger")
        
        create_metric_card("Parameters", "6", "primary")
        create_metric_card("Total Silos", "5", "primary")
    
    # Enhanced results section
    if submitted and active_silos:
        if len(active_silos) == 0:
            st.error("❌ Please select at least one active silo!")
            return
        
        # Enhanced input summary with visualization
        st.markdown("---")
        st.markdown("### 📋 Input Data Summary")
        
        col_table, col_chart = st.columns([1, 1])
        
        with col_table:
            input_df = format_silo_properties(silo_properties)
            st.dataframe(input_df, use_container_width=True, hide_index=True, height=400)
        
        with col_chart:
            # Create radar chart for silo comparison
            radar_fig = create_silo_visualization(silo_properties)
            radar_fig.update_layout(height=400)
            st.plotly_chart(radar_fig, use_container_width=True)
        
        # Enhanced optimization process
        with st.spinner("🔄 Running advanced optimization..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(100):
                progress_bar.progress(i + 1)
                if i < 30:
                    status_text.text("🔍 Initializing optimization parameters...")
                elif i < 60:
                    status_text.text("⚡ Running multi-start optimization...")
                elif i < 90:
                    status_text.text("📊 Evaluating solutions...")
                else:
                    status_text.text("✅ Finalizing results...")
                
                # Small delay to show progress
                import time
                time.sleep(0.01)
            
            try:
                # Call the optimization function
                result, logs = capture_output(
                    optimize_coal_blend_enhanced,
                    silo_properties=silo_properties,
                    n_historical_starts=n_historical_starts,
                    n_random_starts=n_random_starts,
                    active_silos=active_silos,
                    use_enhanced_multi_start=True,
                    verbose=True
                )
                
                progress_bar.empty()
                status_text.empty()
                
                # Enhanced results display
                if result is None:
                    st.error("❌ Optimization function returned None")
                    st.text("Check the debug logs for error details")
                elif not result.get('success', False):
                    st.error("❌ No acceptable solutions found")
                    st.warning("No solutions satisfied at least 3 parameters.")
                    if 'optimization_summary' in result:
                        st.write(f"Total attempts: {result['optimization_summary'].get('total_attempts', 0)}")
                        st.write(f"Successful attempts: {result['optimization_summary'].get('successful_attempts', 0)}")
                else:
                    # Enhanced results display
                    display_optimization_results(result, logs, n_historical_starts, n_random_starts, silo_properties)
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Optimization failed: {str(e)}")
                st.exception(e)

def display_optimization_results(result, logs, n_historical_starts, n_random_starts, silo_properties):
    """Enhanced display of optimization results with beautiful visualizations"""
    
    target_ranges = {
        'BLEND_V.M.%': (24.0, 25.0),
        'BLEND_CSN': (4.5, 6.0),
        'BLEND_Ash%': (0.0, 10.0),
        'BLEND_F.C.%': (60.0, 68.0),
        'BLEND_GM%': (9.0, 11.0),
        'BLEND_I.M.%': (0.0, 1.0)
    }
    
    selected_solution = result['selected_solution']
    
    # Enhanced results header
    st.markdown("---")
    st.markdown("## 🎯 Optimization Results")
    
    # Enhanced summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        satisfaction_ratio = selected_solution['satisfied_count'] / 6
        if satisfaction_ratio == 1.0:
            create_metric_card("Solution Quality", "Perfect", "success")
        elif satisfaction_ratio >= 0.5:
            create_metric_card("Solution Quality", "Good", "warning")
        else:
            create_metric_card("Solution Quality", "Poor", "danger")
    
    with col2:
        create_metric_card("Parameters Met", f"{selected_solution['satisfied_count']}/6", "primary")
        
    with col3:
        success_rate = f"{result['successful_attempts']}/{result['total_attempts']}"
        create_metric_card("Success Rate", success_rate, "primary")
        
    with col4:
        obj_score = f"{selected_solution['objective_value']:.4f}"
        create_metric_card("Objective Score", obj_score, "primary")
    
    st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)
    
    # Enhanced discharge visualization
    st.markdown("### 📊 Optimal Discharge Distribution")
    
    col_chart, col_table = st.columns([2, 1])
    
    with col_chart:
        discharge_fig = create_discharge_chart(selected_solution['optimal_discharges'], result['active_silos'])
        st.plotly_chart(discharge_fig, use_container_width=True)
    
    with col_table:
        discharge_data = []
        for i, discharge in enumerate(selected_solution['optimal_discharges'], 1):
            status = "🟢 Active" if i in result['active_silos'] else "🔴 Inactive"
            discharge_data.append({
                'Silo': f"SILO_{i}",
                'Discharge (%)': f"{discharge:.2f}%",
                'Status': status
            })
        
        discharge_df = pd.DataFrame(discharge_data)
        st.dataframe(discharge_df, use_container_width=True, hide_index=True)
    
    st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)
    
    # Properties table with enhanced formatting
    blend_properties = format_blend_properties(
        selected_solution['evaluation']['predictions'], 
        target_ranges
    )
    
    # Color-code the dataframe
    def color_status(val):
        if val == 'SUCCESS':
            return 'background-color: #28a745; color: white; font-weight: bold'
        elif val == 'VIOLATION':
            return 'background-color: #dc3545; color: white; font-weight: bold'
        return ''
    
    blend_df = pd.DataFrame(blend_properties)
    styled_df = blend_df.style.applymap(color_status, subset=['Status'])
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # Enhanced process logs
    with st.expander("📋 Detailed Process Logs", expanded=False):
        
        # Process summary
        st.markdown("#### 🔍 Optimization Summary")
        
        summary_cols = st.columns(4)
        with summary_cols[0]:
            st.metric("Historical Starts", n_historical_starts)
        with summary_cols[1]:
            st.metric("Random Starts", n_random_starts)
        with summary_cols[2]:
            st.metric("Total Attempts", result['total_attempts'])
        with summary_cols[3]:
            st.metric("Success Rate", f"{(result['successful_attempts']/result['total_attempts']*100):.1f}%")
        
        # Historical data usage
        if result['historical_data_used']:
            st.success("✅ Historical data was successfully utilized")
        else:
            st.warning("⚠️ No historical data available - using random starts only")
        
        # Detailed logs
        st.markdown("#### 📝 Process Log Details")
        st.text_area("", logs, height=300, key="process_logs")
        
        # Additional statistics if available
        if 'optimization_summary' in result:
            st.markdown("#### 📊 Additional Statistics")
            summary = result['optimization_summary']
            
            stats_cols = st.columns(3)
            with stats_cols[0]:
                if 'convergence_time' in summary:
                    st.metric("Convergence Time", f"{summary['convergence_time']:.2f}s")
            with stats_cols[1]:
                if 'best_objective' in summary:
                    st.metric("Best Objective", f"{summary['best_objective']:.6f}")
            with stats_cols[2]:
                if 'iterations' in summary:
                    st.metric("Total Iterations", summary['iterations'])
    
    # Alternative solutions section
    if 'alternative_solutions' in result and result['alternative_solutions']:
        st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)
        st.markdown("### 🔄 Alternative Solutions")
        
        alt_solutions = result['alternative_solutions'][:3]  # Show top 3 alternatives
        
        for i, alt_sol in enumerate(alt_solutions, 1):
            with st.expander(f"Alternative Solution {i} - {alt_sol['satisfied_count']}/6 parameters satisfied"):
                show_alternative_details(alt_sol, target_ranges, result['active_silos'])
    
    # Export results section
    st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)
    st.markdown("### 💾 Export Results")
    
    export_cols = st.columns(3)
    
    with export_cols[0]:
        # Create export data
        export_data = {
            'Optimization Results': {
                'Solution Quality': f"{selected_solution['satisfied_count']}/6 parameters satisfied",
                'Objective Score': selected_solution['objective_value'],
                'Success Rate': f"{result['successful_attempts']}/{result['total_attempts']}"
            },
            'Optimal Discharges': {
                f'SILO_{i}': f"{discharge:.2f}%" 
                for i, discharge in enumerate(selected_solution['optimal_discharges'], 1)
            },
            'Predicted Properties': selected_solution['evaluation']['predictions']
        }
        
        if st.button("📄 Export to JSON", use_container_width=True):
            st.json(export_data)
    
    with export_cols[1]:
        # Create CSV export
        csv_data = pd.DataFrame([
            {'Parameter': 'Solution_Quality', 'Value': f"{selected_solution['satisfied_count']}/6"},
            {'Parameter': 'Objective_Score', 'Value': selected_solution['objective_value']},
            *[{'Parameter': f'Discharge_SILO_{i}', 'Value': f"{discharge:.2f}%"} 
              for i, discharge in enumerate(selected_solution['optimal_discharges'], 1)],
            *[{'Parameter': prop.replace('BLEND_', ''), 'Value': f"{value:.2f}"} 
              for prop, value in selected_solution['evaluation']['predictions'].items()]
        ])
        
        if st.button("📊 Export to CSV", use_container_width=True):
            st.dataframe(csv_data, hide_index=True)
    
    with export_cols[2]:
        if st.button("🔄 Run New Optimization", use_container_width=True):
            st.rerun()

def show_alternative_details(alternative_solution, target_ranges, active_silos):
    """Show detailed view of an alternative solution with enhanced formatting"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**💰 Discharge Values:**")
        for i, discharge in enumerate(alternative_solution['optimal_discharges'], 1):
            status_icon = "🟢" if i in active_silos else "🔴"
            status_text = "Active" if i in active_silos else "Inactive"
            st.write(f"{status_icon} SILO_{i}: {discharge:.2f}% ({status_text})")
    
    with col2:
        st.markdown("**📊 Performance Metrics:**")
        st.write(f"✅ Parameters Satisfied: {alternative_solution['satisfied_count']}/6")
        st.write(f"🎯 Objective Score: {alternative_solution['objective_value']:.6f}")
        st.write(f"🚀 Starting Point: {alternative_solution.get('start_type', 'Unknown')}")
    
    # Alternative blend properties
    st.markdown("**🧪 Predicted Blend Properties:**")
    alt_blend_properties = format_blend_properties(
        alternative_solution['evaluation']['predictions'], 
        target_ranges
    )
    alt_blend_df = pd.DataFrame(alt_blend_properties)
    
    # Apply styling
    def color_alt_status(val):
        if val == 'SUCCESS':
            return 'background-color: #28a745; color: white'
        elif val == 'VIOLATION':
            return 'background-color: #dc3545; color: white'
        return ''
    
    styled_alt_df = alt_blend_df.style.applymap(color_alt_status, subset=['Status'])
    st.dataframe(styled_alt_df, use_container_width=True, hide_index=True)

if __name__ == "__main__":

    main()

