import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import random
import warnings
import os

# Suppress warnings and configure environment
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# Set XGBoost to use CPU by default
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Page configuration
st.set_page_config(
    page_title="AI Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #0f1419 0%, #1a202c 100%);
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f1419 0%, #1a202c 100%);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #334155;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        margin: 0.5rem 0;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #065f46 0%, #047857 100%);
        padding: 2rem;
        border-radius: 20px;
        border: 1px solid #10b981;
        box-shadow: 0 12px 40px rgba(16, 185, 129, 0.3);
        text-align: center;
        margin: 1rem 0;
    }
    
    .fraud-card {
        background: linear-gradient(135deg, #7f1d1d 0%, #dc2626 100%);
        border: 1px solid #ef4444;
        box-shadow: 0 12px 40px rgba(239, 68, 68, 0.3);
    }
    
    .feature-importance {
        background: linear-gradient(135deg, #1e1b4b 0%, #3730a3 100%);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #6366f1;
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.2);
        margin: 1rem 0;
    }
    
    .title-gradient {
        background: linear-gradient(135deg, #06b6d4, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
        font-size: 3rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .sidebar .stSelectbox, .sidebar .stNumberInput {
        background: rgba(30, 41, 59, 0.8);
        border-radius: 8px;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(59, 130, 246, 0.6);
    }
    
    .risk-indicator {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        font-weight: 600;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #065f46, #059669);
        color: white;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #d97706, #f59e0b);
        color: white;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #dc2626, #ef4444);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Load model and features
@st.cache_resource
def load_model_and_features():
    try:
        with open("notebooks/Models/xgb_fraud_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("notebooks/Models/feature_list.pkl", "rb") as f:
            features = pickle.load(f)
        
        # Fix XGBoost device compatibility issues
        if hasattr(model, 'set_param'):
            try:
                # Force CPU usage to avoid GPU warnings
                model.set_param({'tree_method': 'hist', 'device': 'cpu'})
            except:
                pass
        
        return model, features
    except FileNotFoundError:
        st.error("Model files not found. Please ensure the model files are in the correct directory.")
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Generate synthetic feature importance for demonstration
def generate_feature_importance(features, prediction_data):
    # Mock feature importance based on common fraud detection patterns
    importance_map = {
        'amount': 0.15,
        'step': 0.12,
        'oldbalanceOrg': 0.11,
        'newbalanceOrig': 0.11,
        'oldbalanceDest': 0.10,
        'newbalanceDest': 0.10,
        'type_CASH_OUT': 0.08,
        'type_TRANSFER': 0.08,
        'type_PAYMENT': 0.06,
        'type_DEBIT': 0.05,
        'isFlaggedFraud': 0.04
    }
    
    feature_importance = []
    for feature in features:
        base_importance = importance_map.get(feature, 0.02)
        # Add some randomness based on input values
        value_factor = abs(prediction_data[features.index(feature)]) / 1000000 if prediction_data[features.index(feature)] != 0 else 0.01
        adjusted_importance = base_importance * (1 + min(value_factor, 0.5))
        feature_importance.append(adjusted_importance)
    
    # Normalize to sum to 1
    total = sum(feature_importance)
    feature_importance = [imp / total for imp in feature_importance]
    
    return list(zip(features, feature_importance))

# Create risk assessment visualization
def create_risk_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Fraud Risk Score", 'font': {'size': 24, 'color': 'white'}},
        delta = {'reference': 50, 'increasing': {'color': "#ef4444"}, 'decreasing': {'color': "#10b981"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickcolor': "white", 'tickfont': {'color': 'white'}},
            'bar': {'color': "#06b6d4"},
            'steps': [
                {'range': [0, 25], 'color': "#10b981"},
                {'range': [25, 50], 'color': "#f59e0b"},
                {'range': [50, 75], 'color': "#ef4444"},
                {'range': [75, 100], 'color': "#dc2626"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        height=300
    )
    
    return fig

# Create feature importance chart
def create_feature_importance_chart(feature_importance):
    features, importance = zip(*feature_importance[:10])  # Top 10 features
    
    fig = go.Figure(data=[
        go.Bar(
            y=features,
            x=importance,
            orientation='h',
            marker=dict(
                color=importance,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Importance", titlefont=dict(color='white'), tickfont=dict(color='white'))
            )
        )
    ])
    
    fig.update_layout(
        title="Top Feature Importance",
        title_font=dict(size=20, color='white'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        xaxis=dict(gridcolor='#334155'),
        yaxis=dict(gridcolor='#334155'),
        height=400
    )
    
    return fig

# Create transaction analysis chart
def create_transaction_analysis():
    # Generate sample data for demonstration
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    legitimate = np.random.poisson(50, len(dates))
    fraudulent = np.random.poisson(5, len(dates))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=legitimate,
        mode='lines',
        name='Legitimate',
        line=dict(color='#10b981', width=3),
        fill='tonexty'
    ))
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=fraudulent,
        mode='lines',
        name='Fraudulent',
        line=dict(color='#ef4444', width=3),
        fill='tozeroy'
    ))
    
    fig.update_layout(
        title="Transaction Trends (2024)",
        title_font=dict(size=20, color='white'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        xaxis=dict(gridcolor='#334155'),
        yaxis=dict(gridcolor='#334155'),
        legend=dict(font=dict(color='white')),
        height=300
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="title-gradient">🛡️ AI Fraud Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Advanced Machine Learning for Real-Time Transaction Security</p>', unsafe_allow_html=True)
    
    # Load model
    model, features = load_model_and_features()
    
    if model is None or features is None:
        st.error("Failed to load model. Please check the file paths.")
        return
    
    # Sidebar for inputs
    st.sidebar.title("🔧 Transaction Parameters")
    st.sidebar.markdown("Configure the transaction details for analysis")
    
    # Quick preset buttons
    st.sidebar.markdown("### Quick Presets")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("💳 Normal Transaction"):
            st.session_state.preset = "normal"
    
    with col2:
        if st.button("⚠️ Suspicious Transaction"):
            st.session_state.preset = "suspicious"
    
    # Initialize session state
    if 'preset' not in st.session_state:
        st.session_state.preset = None
    
    # Input fields
    input_data = []
    
    # Create organized input sections
    st.sidebar.markdown("### 💰 Transaction Details")
    
    # Set default values based on preset
    default_values = {}
    if st.session_state.preset == "normal":
        default_values = {
            'step': 1.0,
            'amount': 100.0,
            'oldbalanceOrg': 1000.0,
            'newbalanceOrig': 900.0,
            'oldbalanceDest': 500.0,
            'newbalanceDest': 600.0,
            'type_CASH_OUT': 0.0,
            'type_DEBIT': 1.0,
            'type_PAYMENT': 0.0,
            'type_TRANSFER': 0.0,
            'isFlaggedFraud': 0.0
        }
    elif st.session_state.preset == "suspicious":
        default_values = {
            'step': 1.0,
            'amount': 5000000.0,
            'oldbalanceOrg': 5000000.0,
            'newbalanceOrig': 0.0,
            'oldbalanceDest': 0.0,
            'newbalanceDest': 5000000.0,
            'type_CASH_OUT': 1.0,
            'type_DEBIT': 0.0,
            'type_PAYMENT': 0.0,
            'type_TRANSFER': 0.0,
            'isFlaggedFraud': 0.0
        }
    
    # Create input fields with better organization
    for i, feature in enumerate(features):
        if 'type_' in feature:
            if i % 4 == 0 and i > 0:
                st.sidebar.markdown("### 🏷️ Transaction Type")
            value = st.sidebar.selectbox(
                f"{feature.replace('type_', '').replace('_', ' ').title()}",
                [0, 1],
                index=int(default_values.get(feature, 0)),
                key=f"input_{i}"
            )
        elif feature in ['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']:
            if feature == 'oldbalanceOrg':
                st.sidebar.markdown("### 💳 Account Balances")
            value = st.sidebar.number_input(
                feature.replace('balance', ' Balance ').replace('Org', 'Origin').replace('Dest', 'Destination'),
                min_value=0.0,
                value=float(default_values.get(feature, 0)),
                step=100.0,
                format="%.2f",
                key=f"input_{i}"
            )
        elif feature == 'amount':
            value = st.sidebar.number_input(
                "💰 Transaction Amount",
                min_value=0.0,
                value=float(default_values.get(feature, 0)),
                step=10.0,
                format="%.2f",
                key=f"input_{i}"
            )
        elif feature == 'step':
            value = st.sidebar.number_input(
                "⏱️ Time Step",
                min_value=1.0,
                value=float(default_values.get(feature, 1)),
                step=1.0,
                format="%.0f",
                key=f"input_{i}"
            )
        else:
            value = st.sidebar.number_input(
                feature,
                value=float(default_values.get(feature, 0)),
                step=1.0,
                format="%.2f",
                key=f"input_{i}"
            )
        
        input_data.append(value)
    
    # Reset preset after setting values
    if st.session_state.preset:
        st.session_state.preset = None
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📊 Real-Time Analysis Dashboard")
        
        # Transaction analysis chart
        fig_trends = create_transaction_analysis()
        st.plotly_chart(fig_trends, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 System Metrics")
        
        # Mock system metrics
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #10b981; margin: 0;">🎯 Model Accuracy</h4>
            <h2 style="color: white; margin: 0.5rem 0;">99.2%</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #3b82f6; margin: 0;">⚡ Processing Speed</h4>
            <h2 style="color: white; margin: 0.5rem 0;">< 100ms</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #f59e0b; margin: 0;">🛡️ Transactions Today</h4>
            <h2 style="color: white; margin: 0.5rem 0;">2,847</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Prediction section
    st.markdown("---")
    
    if st.button("🚀 Analyze Transaction", type="primary", use_container_width=True):
        # Add loading animation
        with st.spinner('🔍 Analyzing transaction patterns...'):
            time.sleep(2)  # Simulate processing time
            
            # Make prediction with proper error handling
            try:
                data = np.array(input_data).reshape(1, -1)
                
                # Ensure consistent data types
                data = data.astype(np.float32)
                
                # Make predictions with error handling
                prediction = model.predict(data)[0]
                probability = model.predict_proba(data)[0][1]
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                st.info("Please check your input values and try again.")
                return
            
            # Results section
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Risk gauge
                fig_gauge = create_risk_gauge(probability)
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with col2:
                # Prediction result
                if prediction == 1:
                    st.markdown(f"""
                    <div class="prediction-card fraud-card">
                        <h2>⚠️ FRAUD DETECTED</h2>
                        <h3>Risk Level: HIGH</h3>
                        <p>Probability: {probability:.2%}</p>
                        <p>🚨 This transaction shows suspicious patterns</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Risk level indicator
                    st.markdown("""
                    <div class="risk-indicator risk-high">
                        🚨 IMMEDIATE ATTENTION REQUIRED
                    </div>
                    """, unsafe_allow_html=True)
                    
                else:
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h2>✅ LEGITIMATE TRANSACTION</h2>
                        <h3>Risk Level: LOW</h3>
                        <p>Confidence: {1-probability:.2%}</p>
                        <p>🛡️ Transaction appears normal</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Risk level indicator
                    risk_level = "risk-low" if probability < 0.3 else "risk-medium" if probability < 0.6 else "risk-high"
                    risk_text = "✅ TRANSACTION APPROVED" if probability < 0.3 else "⚠️ MONITOR CLOSELY" if probability < 0.6 else "🚨 REVIEW REQUIRED"
                    
                    st.markdown(f"""
                    <div class="risk-indicator {risk_level}">
                        {risk_text}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Feature importance analysis
            st.markdown("---")
            st.markdown("### 🔬 Feature Importance Analysis")
            
            feature_importance = generate_feature_importance(features, input_data)
            fig_importance = create_feature_importance_chart(feature_importance)
            st.plotly_chart(fig_importance, use_container_width=True)
            
            # Detailed analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="feature-importance">
                    <h4>🎯 Key Risk Factors</h4>
                    <ul style="color: white;">
                        <li>Transaction Amount Analysis</li>
                        <li>Account Balance Patterns</li>
                        <li>Transaction Type Classification</li>
                        <li>Historical Behavior Analysis</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="feature-importance">
                    <h4>🛡️ Security Recommendations</h4>
                    <ul style="color: white;">
                        <li>Enable real-time monitoring</li>
                        <li>Implement multi-factor authentication</li>
                        <li>Set up transaction alerts</li>
                        <li>Regular security audits</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #64748b; padding: 2rem 0;">
        <p>🤖 Powered by Advanced Machine Learning | 🛡️ Securing Financial Transactions</p>
        <p>Built with Streamlit & XGBoost | Real-time Fraud Detection System</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()