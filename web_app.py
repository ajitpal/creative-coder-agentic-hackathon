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
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/ajitpal/creative-coder-agentic-hackathon',
        'Report a bug': 'https://github.com/ajitpal/creative-coder-agentic-hackathon/issues',
        'About': 'Intelligent Healthcare Navigator - Your AI-powered healthcare assistant'
    }
)

# Set theme to light mode
st.markdown('''
<script>
    if (window.parent.document.querySelector('section.main')) {
        window.parent.document.querySelector('section.main').style.backgroundColor = '#f8f9fa';
    }
    // Force light theme
    const elements = window.parent.document.querySelectorAll('.stAppToolbar, .stAppToolbar *, .stToolbar, .stToolbar *');
    elements.forEach(el => {
        el.style.backgroundColor = '#f8f9fa';
        el.style.color = '#1a1a1a';
    });
</script>
''', unsafe_allow_html=True)

# Custom CSS for improved readability
st.markdown("""<!-- Force light mode for Streamlit -->
<script>
    // Force light mode by overriding the theme
    localStorage.setItem('theme', 'light');
    // Reload to apply the theme change
    if (document.readyState === 'complete') {
        // Only reload if the page is already loaded
        if (localStorage.getItem('streamlit:themeChanged') !== 'true') {
            localStorage.setItem('streamlit:themeChanged', 'true');
            setTimeout(function() {
                window.location.reload();
            }, 100);
        }
    }
    
    // Add event listener to override toolbar styling after DOM is fully loaded
    document.addEventListener('DOMContentLoaded', function() {
        // Function to apply styles to toolbar
        function styleToolbar() {
            var toolbar = document.querySelector('.stAppToolbar');
            if (toolbar) {
                toolbar.style.backgroundColor = '#f8f9fa';
                toolbar.style.color = '#1a1a1a';
                
                // Style all children
                var allElements = toolbar.querySelectorAll('*');
                allElements.forEach(function(el) {
                    el.style.backgroundColor = '#f8f9fa';
                    el.style.color = '#1a1a1a';
                });
            }
        }
        
        // Initial styling
        styleToolbar();
        
        // Set up a MutationObserver to watch for DOM changes
        var observer = new MutationObserver(styleToolbar);
        observer.observe(document.body, { childList: true, subtree: true });
    });
</script>
<style>
    /* Force light theme */
    :root {
        --background-color: #f8f9fa !important;
        --secondary-background-color: #f8f9fa !important;
        --color-scheme: light !important;
    }
    
    /* Main theme improvements */
    .stApp {
        background-color: #f8f9fa !important;
        color-scheme: light !important;
    }
    
    /* Fix Streamlit toolbar */
    div[data-testid="stToolbar"], 
    div[data-testid="stToolbar"] div, 
    div[data-testid="stToolbar"] span, 
    div[data-testid="stToolbar"] button,
    .stAppToolbar, 
    .stAppToolbar div, 
    .stAppToolbar span, 
    .stAppToolbar button {
        background-color: #f8f9fa !important;
        color: #1a1a1a !important;
    }
    
    /* Fix toolbar button colors */
    div[data-testid="stToolbar"] button svg,
    .stAppToolbar button svg {
        fill: #1a1a1a !important;
    }
    
    /* Fix specific toolbar elements */
    .st-emotion-cache-14vh5up,
    .st-emotion-cache-1j22a0y,
    .st-emotion-cache-70qvj9,
    .st-emotion-cache-scp8yw,
    .st-emotion-cache-1p1m4ay,
    .st-emotion-cache-1gwi02i,
    .st-emotion-cache-1wbqy5l,
    .st-emotion-cache-czk5ss,
    .st-emotion-cache-cqw0tj,
    .st-emotion-cache-1pbsqtx {
        background-color: #f8f9fa !important;
        color: #1a1a1a !important;
    }
    
    /* Override any inline styles */
    [style*="background-color: rgb(14, 17, 23)"],
    [style*="background-color: #0e1117"],
    [style*="background-color:rgb(14,17,23)"],
    [style*="background-color:#0e1117"] {
        background-color: #f8f9fa !important;
        color: #1a1a1a !important;
    }
    
    /* Target the exact toolbar class */
    .st-emotion-cache-14vh5up {
        background-color: #f8f9fa !important;
        border-bottom: 1px solid #e0e0e0 !important;
    }
    
    /* Target all elements in the toolbar */
    .st-emotion-cache-14vh5up * {
        background-color: #f8f9fa !important;
        color: #1a1a1a !important;
    }
    
    /* Target the deploy button */
    .stAppDeployButton button {
        background-color: #2196f3 !important;
        color: white !important;
    }
    
    /* Fix text visibility issues */
    .stApp, .stApp * {
        color: #000000 !important;
    }
    
    /* Ensure all text elements are visible */
    h1, h2, h3, h4, h5, h6, p, span, div, label {
        color: #000000 !important;
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
        color: white !important;
        text-align: center;
        margin: 2rem 0;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        background: linear-gradient(135deg, #2196f3 0%, #1976d2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 3px solid #1976d2;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
        letter-spacing: 0.5px;
        text-transform: none;
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
        color: #000000 !important;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-left: 5px solid #4caf50;
        color: #000000 !important;
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
    
    .info-card h3 {
        color: #1976d2 !important;
        margin-bottom: 0.5rem;
    }
    
    .info-card p {
        color: #333333 !important;
        margin: 0;
    }
    
    /* Warning and success boxes */
    .warning-box {
        background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%);
        border: 2px solid #ff8f00;
        border-radius: 8px;
        padding: 1.2rem;
        margin: 1rem 0;
        color: #e65100 !important;
        font-weight: 500;
    }
    
    .warning-box strong {
        color: #e65100 !important;
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
        box-shadow: 2px 0px 10px rgba(0,0,0,0.08) !important;
        padding: 1rem 0.5rem !important;
    }
    
    /* Make sidebar elements more responsive */
    [data-testid="stSidebar"] .stButton > button {
        width: 100% !important;
        margin: 0.5rem 0 !important;
    }
    
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 {
        color: #1976d2 !important;
        margin-bottom: 1rem !important;
        padding-bottom: 0.5rem !important;
        border-bottom: 1px solid #e0e0e0;
    }
    
    [data-testid="stSidebar"] .stTextInput > div > div > input,
    [data-testid="stSidebar"] .stNumberInput > div > div > input,
    [data-testid="stSidebar"] .stTextArea > div > div > textarea {
        width: 100% !important;
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
        background: linear-gradient(135deg, #2196f3 0%, #1976d2 100%);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-transform: none;
        letter-spacing: 0.5px;
        margin: 0.3rem 0;
        height: auto !important;
        line-height: 1.5 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #42a5f5 0%, #1976d2 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }
    
    /* Input field improvements */
    .stTextInput > div > div > input {
        background-color: white;
        border: 2px solid #e0e0e0;
        border-radius: 8px;
        padding: 0.8rem;
        font-size: 1rem;
        transition: all 0.3s ease;
        color: #000000;
        width: 100% !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #1976d2;
        box-shadow: 0 0 0 3px rgba(25, 118, 210, 0.2);
        transform: translateY(-1px);
    }
    
    /* Make text input container responsive */
    .stTextInput, .stTextArea {
        width: 100% !important;
    }
    
    /* Improve placeholder text */
    .stTextInput > div > div > input::placeholder,
    .stTextArea > div > div > textarea::placeholder {
        color: #9e9e9e !important;
        opacity: 0.8 !important;
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
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        border: 2px solid #e0e0e0;
        font-weight: 600;
        color: #333333 !important;
        transition: all 0.3s ease;
        flex-grow: 1;
        text-align: center;
        min-width: 120px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1976d2 0%, #1565c0 100%);
        color: white !important;
        border-color: #1976d2;
        box-shadow: 0 2px 8px rgba(25, 118, 210, 0.3);
        transform: translateY(-2px);
    }
    
    .stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
        background-color: #e3f2fd;
        color: #1976d2 !important;
        transform: translateY(-1px);
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
        color: #c62828 !important;
        padding: 0.8rem;
        border-radius: 8px;
        font-weight: 700;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(244, 67, 54, 0.2);
    }
    
    .urgency-moderate {
        background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%);
        border: 2px solid #ff8f00;
        color: #e65100 !important;
        padding: 0.8rem;
        border-radius: 8px;
        font-weight: 700;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(255, 143, 0, 0.2);
    }
    
    .urgency-low {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        border: 2px solid #4caf50;
        color: #2e7d32 !important;
        padding: 0.8rem;
        border-radius: 8px;
        font-weight: 700;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(76, 175, 80, 0.2);
    }
    
    /* Footer styling */
    .footer {
        background: linear-gradient(135deg, #f5f5f5 0%, #e0e0e0 100%);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin-top: 2rem;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .footer h4 {
        color: #1976d2 !important;
        margin-bottom: 1rem;
    }
    
    .footer p {
        color: #333333 !important;
        margin-bottom: 0.5rem;
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
    
    async def enhance_response_with_gemini(self, response_text: str, query: str) -> str:
        """Enhance the response with Gemini to make it more user-friendly"""
        if not await self.initialize_agent():
            return response_text
        
        try:
            # Only proceed if we have a Gemini client
            if not st.session_state.agent.gemini_client:
                return response_text
            
            enhancement_prompt = f"""
            Original user query: {query}
            
            Original response: {response_text}
            
            Please enhance this medical information to make it more user-friendly and easier to understand:
            1. Use simpler language while preserving all medical information
            2. Add brief explanations for medical terms in parentheses
            3. Organize information with clear headings and bullet points
            4. Add context about why this information matters to patients
            5. Keep all medical disclaimers intact
            6. Maintain all factual information from the original response
            7. Return ONLY the enhanced response, not your reasoning
            """
            
            # Generate enhanced response
            enhanced_response = await st.session_state.agent.gemini_client.generate_response(
                enhancement_prompt,
                use_functions=False,
                max_tokens=1500
            )
            
            # If enhancement failed, return original
            if not enhanced_response or not enhanced_response.text:
                return response_text
                
            return enhanced_response.text
            
        except Exception as e:
            # Log error but don't fail - return original response
            logger.error(f"Response enhancement failed: {e}")
            return response_text

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
            
            # Create a responsive grid for quick actions
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🩺 Common Conditions", use_container_width=True, key="quick_conditions"):
                    asyncio.run(self.process_text_query("Tell me about common medical conditions"))
            
            with col2:
                if st.button("💊 Drug Safety", use_container_width=True, key="quick_drug_safety"):
                    asyncio.run(self.process_text_query("How can I check if a medication is safe?"))
                    
            # Add more quick actions in a second row
            col3, col4 = st.columns(2)
            with col3:
                if st.button("🚑 Emergency Signs", use_container_width=True, key="quick_emergency"):
                    asyncio.run(self.process_text_query("What are signs of a medical emergency?"))
            
            with col4:
                if st.button("🧠 Mental Health", use_container_width=True, key="quick_mental_health"):
                    asyncio.run(self.process_text_query("How to maintain good mental health?"))
    
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
        
        # Create a responsive grid for example queries
        cols = st.columns(2)
        for i, example in enumerate(examples):
            with cols[i % 2]:
                if st.button(f"📝 {example}", key=f"example_{hash(example)}", use_container_width=True):
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
                
                # Enhance the response with Gemini to make it more user-friendly
                enhanced_text = await self.enhance_response_with_gemini(response.response.response_text, query)
                
                # Add assistant response to chat
                self.add_chat_message("assistant", enhanced_text, {
                    'sources': response.response.sources,
                    'confidence_score': response.response.confidence_score,
                    'processing_time': response.processing_time,
                    'disclaimers': response.response.disclaimers,
                    'urgency_level': response.response.metadata.get('urgency_level') if response.response.metadata else None,
                    'medical_entities': response.response.metadata.get('medical_entities') if response.response.metadata else None
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
        with st.spinner("📄 Analyzing document..."):
            try:
                file_data = uploaded_file.read()
                
                # Process document with agent
                response = await st.session_state.agent.handle_document_upload(
                    file_data, uploaded_file.name, len(file_data)
                )
                
                # Enhance the response with Gemini to make it more user-friendly
                enhanced_text = await self.enhance_response_with_gemini(response.response.response_text, "Analyze this medical document")
                
                # Add assistant response to chat
                self.add_chat_message("assistant", enhanced_text, {
                    'sources': response.response.sources,
                    'confidence_score': response.response.confidence_score,
                    'processing_time': response.processing_time,
                    'disclaimers': response.response.disclaimers,
                    'document_name': uploaded_file.name,
                    'urgency_level': response.response.metadata.get('urgency_level') if response.response.metadata else None,
                    'medical_entities': response.response.metadata.get('medical_entities') if response.response.metadata else None
                })
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Document processing failed: {e}")
                logger.error(f"Document processing error: {e}")
    
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
            logger.error(f"System status check failed: {e}")
            # Set a basic error status so UI shows something
            st.session_state.system_status = {
                'system_healthy': False,
                'error_message': str(e),
                'component_status': {
                    'api_manager': False,
                    'conversation_memory': True,
                    'cache_manager': True
                }
            }
    
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