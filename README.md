# 🏥 Intelligent Healthcare Navigator

An AI-powered healthcare information system that provides safe, accurate, and accessible medical information through multiple interfaces. Built with a safety-first approach using WHO ICD-11, OpenFDA, and Google Gemini APIs.

**🎯 Built for the 2-hour hackathon timeframe with production-ready architecture and comprehensive medical safety features.**

## 🌟 Key Features

### 🩺 Medical Information Access
- **Medical Term Lookup**: Get simplified explanations of medical conditions using WHO ICD-11 data
- **Drug Safety Information**: Access FDA recalls, adverse events, and safety profiles
- **Symptom Analysis**: AI-powered symptom assessment with urgency detection
- **Document Processing**: Analyze medical documents and extract key information

### 🛡️ Safety-First Design
- **Medical Disclaimers**: All responses include appropriate medical disclaimers
- **Emergency Detection**: Automatic detection of urgent medical situations
- **Input Validation**: Comprehensive sanitization and validation of user inputs
- **Allergy Warnings**: Personalized allergy checking and alerts

### 🧠 Intelligent Features
- **Context Awareness**: Remembers conversation history and user preferences
- **Multi-API Integration**: Combines WHO ICD, OpenFDA, and Gemini APIs
- **ReAct Architecture**: Reasoning → Acting → Observation workflow
- **Fallback Mechanisms**: Graceful degradation when APIs are unavailable

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- API keys for Google Gemini, WHO ICD-11, and OpenFDA (optional)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd intelligent-healthcare-navigator

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.template .env
# Edit .env with your API keys
```

### Configuration

Create a `.env` file with your API keys:

```bash
# Required
GEMINI_API_KEY=your_gemini_api_key

# WHO ICD-11 API
WHO_ICD_CLIENT_ID=your_who_client_id
WHO_ICD_CLIENT_SECRET=your_who_client_secret

# OpenFDA API (optional but recommended)
OPENFDA_API_KEY=your_openfda_api_key

# Optional configurations
LOG_LEVEL=INFO
CACHE_TTL=3600
```

### Getting API Keys

#### Google Gemini API
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add to `.env` as `GEMINI_API_KEY`

#### WHO ICD-11 API
1. Register at [WHO ICD API](https://icd.who.int/icdapi)
2. Get client ID and secret
3. Add to `.env` as `WHO_ICD_CLIENT_ID` and `WHO_ICD_CLIENT_SECRET`

#### OpenFDA API (Optional)
1. Register at [OpenFDA](https://open.fda.gov/apis/authentication/)
2. Get API key for higher rate limits
3. Add to `.env` as `OPENFDA_API_KEY`

## 💻 Usage

### Command Line Interface

```bash
# Start interactive CLI
python cli.py

# Direct commands
python cli.py --query "What is diabetes?"
python cli.py --drug "aspirin"
python cli.py --symptom "headache and fever"
python cli.py --upload "medical_report.pdf"
```

#### CLI Commands
- `ask <question>` - Ask a medical question
- `drug <drug_name>` - Get drug information
- `symptom <symptoms>` - Analyze symptoms
- `term <medical_term>` - Look up medical term
- `upload <file_path>` - Upload and analyze document
- `status` - Show system status
- `clear` - Clear conversation history

### Web Interface

```bash
# Start Streamlit web app
streamlit run web_app.py
```

The web interface provides:
- Interactive chat interface
- Document upload functionality
- User preference management
- Conversation history
- System status monitoring

## 🏗️ Architecture

### System Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Input    │───▶│   Agent Core     │───▶│   Responses     │
│  (CLI/Web/API)  │    │  (ReAct Pattern) │    │ (Formatted/Safe)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Core Components                             │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   Query Planner │   Tool Executor │      Memory System          │
│                 │                 │                             │
│ • Query Analysis│ • API Calls     │ • Conversation History      │
│ • Tool Selection│ • Data Processing│ • User Preferences         │
│ • Plan Creation │ • Result Synthesis│ • Caching                 │
└─────────────────┴─────────────────┴─────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    External APIs                                │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   WHO ICD-11    │    OpenFDA      │      Google Gemini          │
│                 │                 │                             │
│ • Medical Terms │ • Drug Recalls  │ • AI Reasoning              │
│ • Disease Codes │ • Adverse Events│ • Text Generation           │
│ • Definitions   │ • Safety Data   │ • Query Analysis            │
└─────────────────┴─────────────────┴─────────────────────────────┘
```

### ReAct Workflow

1. **Reasoning**: Analyze user query and create execution plan
2. **Acting**: Execute plan using appropriate tools and APIs
3. **Observation**: Process results and update memory

## 📚 API Documentation

### Core Classes

#### HealthcareNavigatorAgent
Main orchestrator class that coordinates all system components.

```python
from src.agent import HealthcareNavigatorAgent

agent = HealthcareNavigatorAgent(session_id="user_123")
response = await agent.process_query("What is diabetes?")
```

#### Query Types
- `MEDICAL_TERM`: Medical condition lookup
- `DRUG_INFO`: Drug information and safety
- `SYMPTOMS`: Symptom analysis
- `DOCUMENT_SUMMARY`: Document processing
- `ENTITY_EXTRACTION`: Medical entity extraction

### Example Usage

```python
import asyncio
from src.agent import HealthcareNavigatorAgent

async def main():
    agent = HealthcareNavigatorAgent()
    
    # Process medical query
    response = await agent.process_query("What is hypertension?")
    print(response.response.response_text)
    
    # Upload document
    with open("medical_report.pdf", "rb") as f:
        doc_response = await agent.handle_document_upload(
            f.read(), "medical_report.pdf"
        )
    print(doc_response.response.response_text)

asyncio.run(main())
```

## 🧪 Testing

### Run Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_agent.py

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Test Categories
- **Unit Tests**: Individual component testing
- **Integration Tests**: API integration testing
- **Safety Tests**: Input validation and sanitization
- **Performance Tests**: Response time and throughput

## 🔒 Security & Safety

### Input Validation
- HTML/JavaScript injection prevention
- SQL injection pattern detection
- Command injection protection
- File upload validation

### Medical Safety
- Mandatory medical disclaimers
- Emergency situation detection
- Professional consultation reminders
- Allergy warning system

### Data Privacy
- No persistent storage of medical queries
- Session-based conversation memory
- Configurable data retention policies

## 🚀 Deployment

### Local Development
```bash
# Development server
python cli.py
# or
streamlit run web_app.py
```

### Production Deployment

#### Docker (Recommended)
```bash
# Build image
docker build -t healthcare-navigator .

# Run container
docker run -p 8501:8501 --env-file .env healthcare-navigator
```

#### Cloud Deployment
- **Streamlit Cloud**: Direct deployment from GitHub
- **Heroku**: Use provided Procfile
- **AWS/GCP**: Container deployment
- **Azure**: App Service deployment

### Environment Variables
```bash
# Production settings
ENVIRONMENT=production
LOG_LEVEL=WARNING
CACHE_TTL=7200
MAX_FILE_SIZE=20971520  # 20MB
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600
```

## 📊 Monitoring & Logging

### Logging Levels
- **ERROR**: System errors and failures
- **WARNING**: API issues and fallbacks
- **INFO**: User interactions and processing
- **DEBUG**: Detailed execution information

### Metrics Tracked
- Query processing time
- API response rates
- Error frequencies
- User interaction patterns

### Health Checks
```bash
# System status
python cli.py --status

# API health check
curl http://localhost:8501/health
```

## 🤝 Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run code formatting
black src/ tests/
flake8 src/ tests/
```

### Code Standards
- **Black** for code formatting
- **Flake8** for linting
- **Type hints** for all functions
- **Docstrings** for all classes and methods
- **Unit tests** for all new features

### Pull Request Process
1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit pull request with description

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Medical Disclaimer

**IMPORTANT**: This system is for informational and educational purposes only. It is not intended to be a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare providers with any questions regarding medical conditions. In case of medical emergency, contact emergency services immediately.

## 🆘 Support

### Documentation
- [Demo Guide](DEMO.md) - Quick demonstration guide
- [Architecture](ARCHITECTURE.md) - Detailed system architecture
- [API Reference](docs/api.md) - Complete API documentation

### Getting Help
- **Issues**: Report bugs and feature requests on GitHub
- **Discussions**: Community discussions and Q&A
- **Email**: Contact maintainers for urgent issues

### FAQ

**Q: Can this system diagnose medical conditions?**
A: No, this system provides educational information only and cannot diagnose medical conditions. Always consult healthcare professionals.

**Q: How accurate is the medical information?**
A: Information comes from authoritative sources (WHO ICD-11, FDA) but should be verified with healthcare providers.

**Q: Is my data stored or shared?**
A: Conversations are stored temporarily for session continuity but are not permanently stored or shared.

**Q: What file types are supported for document upload?**
A: PDF, TXT, DOC, DOCX, and RTF files up to 10MB.

**Q: How do I get API keys?**
A: Follow the links in the Configuration section above for each API provider.

---

Built with ❤️ for accessible healthcare information