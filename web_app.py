"""
Streamlit Web Interface for Intelligent Healthcare Navigator
Provides interactive web-based healthcare navigation
"""

import streamlit as st
import asyncio
import sys
import os
from typing import Dict, Any, List
import time
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.agent import HealthcareNavigatorAgent
from src.config import Config
from src.utils import setup_logging

logger = setup_logging()

# Page configuration
st.set_page_config(
    page_title="Healthcare Navigator",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for improved readability
st.markdown("""
<style>
    /* Main theme improvements */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Fix text visibility issues */
    .stApp, .stApp * {
        color: #1a1a1a !important;
    }
    
    /* Ensure all text elements are visible */
    h1, h2, h3, h4, h5, h6, p, span, div, label {
        color: #1a1a1a !important;
    }
    
    /* Streamlit specific text fixes */
    .stMarkdown, .stMarkdown * {
        color: #1a1a1a !important;
    }
    
    .stText, .stText * {
        color: #1a1a1a !important;
    }
    
    /* Header text fixes */
    .stHeader, .stSubheader {
        color: #1a1a1a !important;
    }
    
    /* Input labels and text */
    .stTextInput label, .stTextArea label, .stSelectbox label, .stNumberInput label {
        color: #1a1a1a !important;
        font-weight: 500 !important;
    }
    
    /* Button text should remain white on colored backgrounds */
    .stButton > button {
        color: white !important;
    }
    
    /* Tab text */
    .stTabs [data-baseweb="tab"] {
        color: #1a1a1a !important;
    }
    
    .stTabs [aria-selected="true"] {
        color: white !important;
    }
    
    /* Expander text */
    .streamlit-expanderHeader {
        color: #1a1a1a !important;
    }
    
    /* Sidebar text */
    .css-1d391kg * {
        color: #1a1a1a !important;
    }
    
    /* File uploader fixes */
    .stFileUploader {
        background-color: white !important;
        border: 2px solid #e0e0e0 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    .stFileUploader label {
        color: #1a1a1a !important;
        font-weight: 500 !important;
    }
    
    .stFileUploader > div {
        background-color: white !important;
        color: #1a1a1a !important;
    }
    
    .stFileUploader * {
        color: #1a1a1a !important;
        background-color: transparent !important;
    }
    
    /* Dropdown/Selectbox fixes */
    .stSelectbox {
        background-color: white !important;
    }
    
    .stSelectbox label {
        color: #1a1a1a !important;
        font-weight: 500 !important;
    }
    
    .stSelectbox > div > div {
        background-color: white !important;
        color: #1a1a1a !important;
        border: 2px solid #e0e0e0 !important;
        border-radius: 8px !important;
    }
    
    .stSelectbox > div > div > div {
        color: #1a1a1a !important;
        background-color: white !important;
    }
    
    /* Dropdown menu styling */
    [data-baseweb="select"] {
        background-color: white !important;
    }
    
    [data-baseweb="select"] > div {
        background-color: white !important;
        color: #1a1a1a !important;
        border: 2px solid #e0e0e0 !important;
    }
    
    /* Dropdown options */
    [role="listbox"] {
        background-color: white !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    }
    
    [role="option"] {
        background-color: white !important;
        color: #1a1a1a !important;
        padding: 0.5rem 1rem !important;
    }
    
    [role="option"]:hover {
        background-color: #f0f0f0 !important;
        color: #1a1a1a !important;
    }
    
    [aria-selected="true"][role="option"] {
        background-color: #1976d2 !important;
        color: white !important;
    }
    
    /* Text area fixes */
    .stTextArea > div > div > textarea {
        background-color: white !important;
        color: #1a1a1a !important;
        border: 2px solid #e0e0e0 !important;
        border-radius: 8px !important;
    }
    
    .stTextArea label {
        color: #1a1a1a !important;
        font-weight: 500 !important;
    }
    
    /* Number input fixes */
    .stNumberInput > div > div > input {
        background-color: white !important;
        color: #1a1a1a !important;
        border: 2px solid #e0e0e0 !important;
        border-radius: 8px !important;
    }
    
    .stNumberInput label {
        color: #1a1a1a !important;
        font-weight: 500 !important;
    }
    
    /* Header styling - Make it highly visible */
    .main-header {
        font-size: 3rem;
        color: #1a1a1a !important;
        text-align: center;
        margin: 2rem 0;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        background: linear-gradient(135deg, #ffffff 0%, #f0f0f0 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 3px solid #1976d2;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    }
    
    /* Chat message improvements */
    .chat-message {
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        font-size: 1rem;
        line-height: 1.6;
    }
    
    .user-message {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 5px solid #1976d2;
        color: #1565c0;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
        border-left: 5px solid #7b1fa2;
        color: #4a148c;
    }
    
    /* Info boxes with better contrast */
    .info-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }
    
    .info-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Warning and success boxes */
    .warning-box {
        background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%);
        border: 2px solid #ff8f00;
        border-radius: 8px;
        padding: 1.2rem;
        margin: 1rem 0;
        color: #e65100;
        font-weight: 500;
    }
    
    .success-box {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        border: 2px solid #4caf50;
        border-radius: 8px;
        padding: 1.2rem;
        margin: 1rem 0;
        color: #2e7d32;
        font-weight: 500;
    }
    
    .error-box {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border: 2px solid #f44336;
        border-radius: 8px;
        padding: 1.2rem;
        margin: 1rem 0;
        color: #c62828;
        font-weight: 500;
    }
    
    /* Sidebar improvements - Fix readability with multiple selectors */
    .css-1d391kg, .css-1lcbmhc, .css-17eq0hr, section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 2px solid #e0e0e0 !important;
    }
    
    /* Sidebar specific text styling - comprehensive selectors */
    .css-1d391kg *, .css-1lcbmhc *, .css-17eq0hr *, section[data-testid="stSidebar"] * {
        color: #1a1a1a !important;
        background-color: transparent !important;
    }
    
    /* Sidebar headers - multiple selectors */
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3,
    .css-1lcbmhc h1, .css-1lcbmhc h2, .css-1lcbmhc h3,
    .css-17eq0hr h1, .css-17eq0hr h2, .css-17eq0hr h3,
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        color: #1976d2 !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar input labels - comprehensive */
    .css-1d391kg label, .css-1lcbmhc label, .css-17eq0hr label, section[data-testid="stSidebar"] label {
        color: #333 !important;
        font-weight: 500 !important;
    }
    
    /* Sidebar text elements - comprehensive */
    .css-1d391kg p, .css-1d391kg span, .css-1d391kg div,
    .css-1lcbmhc p, .css-1lcbmhc span, .css-1lcbmhc div,
    .css-17eq0hr p, .css-17eq0hr span, .css-17eq0hr div,
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] div {
        color: #1a1a1a !important;
    }
    
    /* Sidebar buttons - comprehensive selectors */
    .css-1d391kg .stButton > button, .css-1lcbmhc .stButton > button, 
    .css-17eq0hr .stButton > button, section[data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #1976d2 0%, #1565c0 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: 500 !important;
    }
    
    /* Sidebar input fields - comprehensive */
    .css-1d391kg .stTextInput > div > div > input,
    .css-1d391kg .stTextArea > div > div > textarea,
    .css-1d391kg .stNumberInput > div > div > input,
    .css-1lcbmhc .stTextInput > div > div > input,
    .css-1lcbmhc .stTextArea > div > div > textarea,
    .css-1lcbmhc .stNumberInput > div > div > input,
    .css-17eq0hr .stTextInput > div > div > input,
    .css-17eq0hr .stTextArea > div > div > textarea,
    .css-17eq0hr .stNumberInput > div > div > input,
    section[data-testid="stSidebar"] .stTextInput > div > div > input,
    section[data-testid="stSidebar"] .stTextArea > div > div > textarea,
    section[data-testid="stSidebar"] .stNumberInput > div > div > input {
        background-color: white !important;
        color: #1a1a1a !important;
        border: 2px solid #e0e0e0 !important;
        border-radius: 8px !important;
    }
    
    /* Additional sidebar text fixes */
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stText {
        color: #1a1a1a !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #1976d2 0%, #1565c0 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #1565c0 0%, #0d47a1 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Input field improvements */
    .stTextInput > div > div > input {
        background-color: white;
        border: 2px solid #e0e0e0;
        border-radius: 8px;
        padding: 0.8rem;
        font-size: 1rem;
        transition: border-color 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #1976d2;
        box-shadow: 0 0 0 3px rgba(25, 118, 210, 0.1);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f5f5f5;
        border-radius: 8px;
        font-weight: 500;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        border: 2px solid #e0e0e0;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1976d2 0%, #1565c0 100%);
        color: white;
        border-color: #1976d2;
    }
    
    /* Metric styling */
    .metric-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1976d2;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        font-weight: 500;
    }
    
    /* Urgency indicators */
    .urgency-high {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border: 2px solid #f44336;
        color: #c62828;
        padding: 0.8rem;
        border-radius: 8px;
        font-weight: 600;
        text-align: center;
        margin: 1rem 0;
    }
    
    .urgency-moderate {
        background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%);
        border: 2px solid #ff8f00;
        color: #e65100;
        padding: 0.8rem;
        border-radius: 8px;
        font-weight: 600;
        text-align: center;
        margin: 1rem 0;
    }
    
    .urgency-low {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        border: 2px solid #4caf50;
        color: #2e7d32;
        padding: 0.8rem;
        border-radius: 8px;
        font-weight: 600;
        text-align: center;
        margin: 1rem 0;
    }
    
    /* Footer styling */
    .footer {
        background: linear-gradient(135deg, #f5f5f5 0%, #e0e0e0 100%);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin-top: 2rem;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

class HealthcareWebApp:
    """Streamlit web application for healthcare navigation"""
    
    def __init__(self):
        """Initialize web app"""
        self.initialize_session_state()
    
    def format_medical_response(self, response_text: str) -> str:
        """Format medical response text for better readability with markdown support"""
        import re
        
        # Clean up duplicate disclaimers first
        formatted_text = self.clean_duplicate_disclaimers(response_text)
        
        # Convert markdown formatting to HTML
        formatted_text = self.convert_markdown_to_html(formatted_text)
        
        # Highlight important medical terms
        medical_keywords = [
            'URGENT', 'EMERGENCY', 'IMMEDIATE', 'SERIOUS', 'WARNING', 'CAUTION',
            'CONTRAINDICATED', 'SIDE EFFECTS', 'ADVERSE REACTIONS', 'DOSAGE',
            'PRESCRIPTION', 'MEDICATION', 'TREATMENT', 'DIAGNOSIS', 'SYMPTOMS'
        ]
        
        for keyword in medical_keywords:
            formatted_text = formatted_text.replace(
                keyword, 
                f'<strong style="color: #d32f2f;">{keyword}</strong>'
            )
        
        return formatted_text
    
    def clean_duplicate_disclaimers(self, text: str) -> str:
        """Remove duplicate medical disclaimers"""
        import re
        
        # Common disclaimer patterns
        disclaimer_patterns = [
            r'⚠️ MEDICAL DISCLAIMER:.*?immediately\.',
            r'⚠️ This information is for educational purposes only.*?medical advice\.',
            r'⚠️ This analysis does not constitute medical diagnosis.*?medical concerns\.',
            r'⚠️ This system does not provide medical diagnoses.*?medical evaluation\.',
            r'\*Disclaimer:.*?treatment\.\*'
        ]
        
        # Remove duplicate disclaimers
        for pattern in disclaimer_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            if len(matches) > 1:
                # Keep only the first occurrence
                for i in range(1, len(matches)):
                    text = text.replace(matches[i], '', 1)
        
        return text
    
    def convert_markdown_to_html(self, text: str) -> str:
        """Convert basic markdown formatting to HTML"""
        import re
        
        # Convert headers
        text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
        
        # Convert bullet points
        text = re.sub(r'^\* (.*?)$', r'• \1', text, flags=re.MULTILINE)
        text = re.sub(r'^\*\* (.*?)$', r'<strong>• \1</strong>', text, flags=re.MULTILINE)
        
        # Convert line breaks
        text = text.replace('\n\n', '<br><br>')
        text = text.replace('\n', '<br>')
        
        # Format sections with better styling
        text = re.sub(r'<strong>(Possible Causes|Severity Assessment|Self-Care Recommendations|When to Seek Care|Red Flags)</strong>', 
                     r'<h4 style="color: #1976d2; margin-top: 1rem; margin-bottom: 0.5rem;">\1</h4>', text)
        
        # Style bullet points
        text = re.sub(r'• (.*?)<br>', r'<div style="margin-left: 1rem; margin-bottom: 0.3rem;">• \1</div>', text)
        
        return text
    
    def format_entity_highlights(self, text: str, entities: List[Dict]) -> str:
        """Highlight medical entities in text"""
        if not entities:
            return text
        
        # Sort entities by position to avoid overlap issues
        sorted_entities = sorted(entities, key=lambda x: x.get('start', 0), reverse=True)
        
        for entity in sorted_entities:
            entity_text = entity.get('text', '')
            entity_type = entity.get('entity_type', 'unknown')
            
            # Color coding for different entity types
            color_map = {
                'disease': '#e53e3e',
                'drug': '#3182ce', 
                'symptom': '#d69e2e',
                'procedure': '#38a169'
            }
            
            color = color_map.get(entity_type, '#666')
            
            if entity_text in text:
                highlighted = f'<span style="background-color: {color}20; color: {color}; padding: 2px 4px; border-radius: 4px; font-weight: 500;">{entity_text}</span>'
                text = text.replace(entity_text, highlighted)
        
        return text
    
    def initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'agent' not in st.session_state:
            st.session_state.agent = None
        
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        if 'session_id' not in st.session_state:
            st.session_state.session_id = f"web_{int(time.time())}"
        
        if 'system_status' not in st.session_state:
            st.session_state.system_status = None
        
        if 'user_preferences' not in st.session_state:
            st.session_state.user_preferences = {}
    
    async def initialize_agent(self):
        """Initialize healthcare agent"""
        if st.session_state.agent is None:
            try:
                with st.spinner("Initializing Healthcare Navigator..."):
                    st.session_state.agent = HealthcareNavigatorAgent(st.session_state.session_id)
                st.success("✅ Healthcare Navigator initialized successfully!")
                return True
            except Exception as e:
                st.error(f"❌ Failed to initialize Healthcare Navigator: {e}")
                logger.error(f"Web app initialization failed: {e}")
                return False
        return True
    
    def render_header(self):
        """Render application header"""
        st.markdown('<h1 class="main-header">🏥 Intelligent Healthcare Navigator</h1>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Quick info cards with improved styling
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="info-card">
                <h3 style="color: #1976d2; margin-bottom: 0.5rem;">💊 Drug Information</h3>
                <p style="margin: 0; color: #666;">Get comprehensive drug safety data, recalls, and adverse events</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="info-card">
                <h3 style="color: #1976d2; margin-bottom: 0.5rem;">🩺 Medical Terms</h3>
                <p style="margin: 0; color: #666;">Understand medical conditions in plain language</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="info-card">
                <h3 style="color: #1976d2; margin-bottom: 0.5rem;">📄 Document Analysis</h3>
                <p style="margin: 0; color: #666;">Analyze medical documents and prescriptions</p>
            </div>
            """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar with controls and information"""
        with st.sidebar:
            st.header("🔧 Controls")
            
            # System status
            if st.button("🔍 Check System Status"):
                asyncio.run(self.check_system_status())
            
            if st.session_state.system_status:
                status = st.session_state.system_status
                if status.get('system_healthy'):
                    st.success("✅ System Healthy")
                else:
                    st.error("❌ System Issues Detected")
            
            st.markdown("---")
            
            # User preferences
            st.header("👤 User Preferences")
            
            age = st.number_input("Age", min_value=0, max_value=120, value=st.session_state.user_preferences.get('age', 30))
            if age != st.session_state.user_preferences.get('age'):
                st.session_state.user_preferences['age'] = age
                if st.session_state.agent:
                    st.session_state.agent.set_user_preference('age', age)
            
            allergies = st.text_area("Known Allergies (one per line)", 
                                   value="\\n".join(st.session_state.user_preferences.get('allergies', [])))
            allergy_list = [a.strip() for a in allergies.split('\\n') if a.strip()]
            if allergy_list != st.session_state.user_preferences.get('allergies', []):
                st.session_state.user_preferences['allergies'] = allergy_list
                if st.session_state.agent:
                    st.session_state.agent.set_user_preference('allergies', allergy_list)
            
            medical_history = st.text_area("Medical History (one per line)",
                                         value="\\n".join(st.session_state.user_preferences.get('medical_history', [])))
            history_list = [h.strip() for h in medical_history.split('\\n') if h.strip()]
            if history_list != st.session_state.user_preferences.get('medical_history', []):
                st.session_state.user_preferences['medical_history'] = history_list
                if st.session_state.agent:
                    st.session_state.agent.set_user_preference('medical_history', history_list)
            
            st.markdown("---")
            
            # Session management
            st.header("💬 Session")
            st.text(f"Session ID: {st.session_state.session_id}")
            
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                if st.session_state.agent:
                    st.session_state.agent.clear_conversation_history()
                st.success("Chat history cleared!")
            
            st.markdown("---")
            
            # Quick actions
            st.header("⚡ Quick Actions")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🩺 Common Conditions"):
                    st.session_state.quick_query = "Tell me about common medical conditions"
            
            with col2:
                if st.button("💊 Drug Safety"):
                    st.session_state.quick_query = "How can I check if a medication is safe?"
    
    def render_chat_interface(self):
        """Render main chat interface"""
        st.header("💬 Chat with Healthcare Navigator")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                self.render_chat_message(message)
        
        # Query input methods
        tab1, tab2, tab3 = st.tabs(["💬 Text Query", "📄 Document Upload", "🔍 Specific Lookup"])
        
        with tab1:
            self.render_text_query_interface()
        
        with tab2:
            self.render_document_upload_interface()
        
        with tab3:
            self.render_specific_lookup_interface()
    
    def render_chat_message(self, message: Dict[str, Any]):
        """Render a single chat message"""
        if message['type'] == 'user':
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👤 You:</strong> {message['content']}
                <br><small>{message['timestamp']}</small>
            </div>
            """, unsafe_allow_html=True)
        
        elif message['type'] == 'assistant':
            # Format the response text for better readability
            formatted_content = self.format_medical_response(message['content'])
            
            # Apply entity highlighting if entities are available
            if 'metadata' in message and 'medical_entities' in message['metadata']:
                formatted_content = self.format_entity_highlights(
                    formatted_content, 
                    message['metadata']['medical_entities']
                )
            
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>🏥 Healthcare Navigator:</strong><br>
                {formatted_content}
                <br><small>{message['timestamp']}</small>
            </div>
            """, unsafe_allow_html=True)
            
            # Show additional info if available
            if 'metadata' in message:
                metadata = message['metadata']
                
                with st.expander("📊 Response Details"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if 'sources' in metadata:
                            st.write("**Sources:**")
                            for source in metadata['sources']:
                                st.write(f"• {source}")
                        
                        if 'confidence_score' in metadata:
                            st.write(f"**Confidence:** {metadata['confidence_score']:.1%}")
                    
                    with col2:
                        if 'processing_time' in metadata:
                            st.write(f"**Processing Time:** {metadata['processing_time']:.2f}s")
                        
                        if 'urgency_level' in metadata:
                            urgency = metadata['urgency_level']
                            if urgency and urgency.lower() == 'high':
                                st.markdown(f'<div class="urgency-high">🚨 HIGH URGENCY - Seek immediate medical attention</div>', unsafe_allow_html=True)
                            elif urgency and urgency.lower() == 'moderate':
                                st.markdown(f'<div class="urgency-moderate">⚠️ MODERATE URGENCY - Consider consulting a healthcare provider</div>', unsafe_allow_html=True)
                            elif urgency and urgency.lower() == 'low':
                                st.markdown(f'<div class="urgency-low">ℹ️ LOW URGENCY - General information provided</div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="urgency-low">ℹ️ INFORMATIONAL - General healthcare guidance</div>', unsafe_allow_html=True)
                
                # Show disclaimers
                if 'disclaimers' in metadata:
                    st.markdown("""
                    <div class="warning-box">
                        <strong>⚠️ Important Medical Disclaimers:</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    for disclaimer in metadata['disclaimers']:
                        st.write(f"• {disclaimer}")
    
    def render_text_query_interface(self):
        """Render text query interface"""
        # Handle quick query
        if hasattr(st.session_state, 'quick_query'):
            query = st.session_state.quick_query
            del st.session_state.quick_query
        else:
            query = st.text_input("Ask a medical question:", placeholder="e.g., What is diabetes? Tell me about aspirin side effects.")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("🚀 Send Query", disabled=not query):
                asyncio.run(self.process_text_query(query))
        
        with col2:
            if st.button("🔄 Clear Input"):
                st.rerun()
        
        # Example queries
        st.markdown("**💡 Example queries:**")
        examples = [
            "What is hypertension?",
            "Tell me about aspirin side effects",
            "I have a headache and fever, what should I do?",
            "Is ibuprofen safe during pregnancy?"
        ]
        
        for example in examples:
            if st.button(f"📝 {example}", key=f"example_{hash(example)}"):
                asyncio.run(self.process_text_query(example))
    
    def render_document_upload_interface(self):
        """Render document upload interface"""
        st.write("Upload medical documents for analysis (PDF, TXT, DOC)")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'txt', 'doc', 'docx'],
            help="Upload medical reports, prescriptions, or other healthcare documents"
        )
        
        if uploaded_file is not None:
            st.write(f"**File:** {uploaded_file.name}")
            st.write(f"**Size:** {uploaded_file.size:,} bytes")
            
            if st.button("📄 Analyze Document"):
                asyncio.run(self.process_document_upload(uploaded_file))
    
    def render_specific_lookup_interface(self):
        """Render specific lookup interface"""
        lookup_type = st.selectbox(
            "Select lookup type:",
            ["Medical Term", "Drug Information", "Symptom Analysis"]
        )
        
        if lookup_type == "Medical Term":
            term = st.text_input("Enter medical term:", placeholder="e.g., diabetes, hypertension")
            if st.button("🔍 Look up Term") and term:
                asyncio.run(self.process_text_query(f"What is {term}?"))
        
        elif lookup_type == "Drug Information":
            drug = st.text_input("Enter drug name:", placeholder="e.g., aspirin, ibuprofen")
            if st.button("💊 Get Drug Info") and drug:
                asyncio.run(self.process_text_query(f"Tell me about the drug {drug}"))
        
        elif lookup_type == "Symptom Analysis":
            symptoms = st.text_area("Describe your symptoms:", placeholder="e.g., headache, fever, nausea")
            if st.button("🩺 Analyze Symptoms") and symptoms:
                asyncio.run(self.process_text_query(f"I have these symptoms: {symptoms}"))
    
    async def process_text_query(self, query: str):
        """Process text query"""
        if not await self.initialize_agent():
            return
        
        # Add user message to chat
        self.add_chat_message("user", query)
        
        # Show processing indicator
        with st.spinner("🤔 Processing your query..."):
            try:
                # Prepare context with user preferences
                context = {
                    'user_age': st.session_state.user_preferences.get('age'),
                    'user_allergies': st.session_state.user_preferences.get('allergies', []),
                    'medical_history': st.session_state.user_preferences.get('medical_history', [])
                }
                
                response = await st.session_state.agent.process_query(query, context)
                
                # Add assistant response to chat
                self.add_chat_message("assistant", response.response.response_text, {
                    'sources': response.response.sources,
                    'confidence_score': response.response.confidence_score,
                    'processing_time': response.processing_time,
                    'disclaimers': response.response.disclaimers,
                    'urgency_level': response.response.metadata.get('urgency_level') if response.response.metadata else None
                })
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Query processing failed: {e}")
                logger.error(f"Web query processing error: {e}")
    
    async def process_document_upload(self, uploaded_file):
        """Process document upload"""
        if not await self.initialize_agent():
            return
        
        # Add user message to chat
        self.add_chat_message("user", f"📄 Uploaded document: {uploaded_file.name}")
        
        # Show processing indicator
        with st.spinner("📄 Processing document..."):
            try:
                file_data = uploaded_file.read()
                
                response = await st.session_state.agent.handle_document_upload(
                    file_data, uploaded_file.name, len(file_data)
                )
                
                # Add assistant response to chat
                self.add_chat_message("assistant", response.response.response_text, {
                    'sources': response.response.sources,
                    'confidence_score': response.response.confidence_score,
                    'processing_time': response.processing_time,
                    'disclaimers': response.response.disclaimers
                })
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Document processing failed: {e}")
                logger.error(f"Web document processing error: {e}")
    
    async def check_system_status(self):
        """Check system status"""
        if not await self.initialize_agent():
            return
        
        try:
            with st.spinner("Checking system status..."):
                status = await st.session_state.agent.get_system_status()
                st.session_state.system_status = status
                st.rerun()
                
        except Exception as e:
            st.error(f"❌ Status check failed: {e}")
    
    def add_chat_message(self, message_type: str, content: str, metadata: Dict[str, Any] = None):
        """Add message to chat history"""
        message = {
            'type': message_type,
            'content': content,
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'metadata': metadata or {}
        }
        st.session_state.chat_history.append(message)
    
    def run(self):
        """Run the web application"""
        self.render_header()
        self.render_sidebar()
        self.render_chat_interface()
        
        # Footer with improved styling
        st.markdown("---")
        st.markdown("""
        <div class="footer">
            <h4 style="color: #1976d2; margin-bottom: 1rem;">🏥 Intelligent Healthcare Navigator</h4>
            <p style="margin-bottom: 0.5rem; color: #666;">Built for Healthcare Information Access</p>
            <div class="warning-box" style="margin: 1rem auto; max-width: 600px;">
                <strong>⚠️ Important Medical Disclaimer:</strong><br>
                This tool provides information only and is not a substitute for professional medical advice, diagnosis, or treatment. 
                Always consult healthcare professionals for medical decisions.
            </div>
            <p style="margin: 0; color: #999; font-size: 0.9rem;">
                Emergency? Call 911 (US) or your local emergency number immediately.
            </p>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main entry point for Streamlit app"""
    app = HealthcareWebApp()
    app.run()

if __name__ == "__main__":
    main()