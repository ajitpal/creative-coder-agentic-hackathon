# Healthcare Navigator Demo Guide

## Quick Start Demo

### 1. Setup (2 minutes)

```bash
# Clone and setup
git clone <repository>
cd intelligent-healthcare-navigator

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.template .env
# Edit .env with your API keys
```

### 2. CLI Demo (3 minutes)

```bash
# Start CLI
python cli.py

# Try these commands:
ask "What is diabetes?"
drug "aspirin"
symptom "headache and fever"
upload "sample_report.pdf"
```

### 3. Web Interface Demo (5 minutes)

```bash
# Start web app
streamlit run web_app.py
```

**Demo Flow:**
1. **Medical Term Lookup**: "What is hypertension?"
2. **Drug Safety Check**: "Tell me about ibuprofen safety"
3. **Symptom Analysis**: "I have chest pain and shortness of breath"
4. **Document Upload**: Upload a medical report
5. **User Preferences**: Set age, allergies, medical history

## Key Features to Demonstrate

### 🩺 Medical Term Explanations
- **Input**: "What is atrial fibrillation?"
- **Shows**: WHO ICD data + AI simplification
- **Highlights**: Technical vs. simplified explanations

### 💊 Drug Information & Safety
- **Input**: "Tell me about warfarin"
- **Shows**: FDA recalls, adverse events, safety profile
- **Highlights**: Allergy warnings, safety assessments

### 🔍 Symptom Analysis
- **Input**: "I have severe chest pain and difficulty breathing"
- **Shows**: Urgency detection, care recommendations
- **Highlights**: Emergency detection, medical disclaimers

### 📄 Document Processing
- **Input**: Upload medical report PDF
- **Shows**: Summary, key findings, entity extraction
- **Highlights**: Medical entity recognition

### 🧠 Context Awareness
- **Shows**: Conversation memory, user preferences
- **Highlights**: Personalized responses based on age/allergies

## Demo Script (10 minutes)

### Opening (1 min)
"Healthcare Navigator is an AI-powered system that helps users understand medical information safely and accurately."

### Core Capabilities (6 mins)

**1. Medical Knowledge (2 mins)**
```
CLI: ask "What is Type 2 diabetes?"
```
- Shows WHO ICD integration
- Demonstrates AI simplification
- Highlights medical disclaimers

**2. Drug Safety (2 mins)**
```
Web: Search "aspirin" with allergy to "NSAIDs"
```
- Shows FDA data integration
- Demonstrates allergy warnings
- Highlights safety assessments

**3. Symptom Intelligence (2 mins)**
```
CLI: symptom "severe chest pain, shortness of breath, sweating"
```
- Shows urgency detection
- Demonstrates emergency recommendations
- Highlights responsible AI responses

### Advanced Features (2 mins)

**4. Document Analysis**
```
Web: Upload medical report
```
- Shows document processing
- Demonstrates entity extraction
- Highlights key findings

### Closing (1 min)
"Healthcare Navigator combines multiple medical APIs with responsible AI to provide accurate, safe healthcare information."

## Technical Highlights

### Architecture
- **ReAct Pattern**: Reasoning → Acting → Observation
- **Multi-API Integration**: WHO ICD, OpenFDA, Google Gemini
- **Safety-First Design**: Disclaimers, validation, error handling

### APIs Demonstrated
- **WHO ICD-11**: Medical term definitions
- **OpenFDA**: Drug recalls, adverse events
- **Google Gemini**: AI reasoning and simplification

### Safety Features
- Input sanitization and validation
- Medical disclaimers on all responses
- Emergency detection and warnings
- Allergy checking and alerts

## Sample Queries for Demo

### Medical Terms
- "What is myocardial infarction?"
- "Explain rheumatoid arthritis"
- "What does hypertension mean?"

### Drug Information
- "Is acetaminophen safe during pregnancy?"
- "Tell me about metformin side effects"
- "What recalls exist for insulin?"

### Symptoms (Use Carefully)
- "I have a persistent cough and fever"
- "What could cause severe headaches?"
- "I'm experiencing dizziness and nausea"

### Emergency Examples (Shows Urgency Detection)
- "I have severe chest pain and can't breathe"
- "Someone is unconscious and not responding"
- "I think I'm having a stroke"

## Demo Environment Setup

### Required API Keys
```bash
# .env file
GEMINI_API_KEY=your_gemini_key
WHO_ICD_CLIENT_ID=your_who_client_id
WHO_ICD_CLIENT_SECRET=your_who_secret
OPENFDA_API_KEY=your_fda_key  # Optional but recommended
```

### Sample Files for Upload
- Create sample medical reports (PDF/TXT)
- Include various medical entities
- Test different document types

## Troubleshooting Demo Issues

### Common Issues
1. **API Rate Limits**: Use demo keys with higher limits
2. **Slow Responses**: Pre-warm APIs before demo
3. **Network Issues**: Have offline fallback examples

### Fallback Strategies
- Use cached responses for key demos
- Have screenshots ready
- Prepare offline CLI examples

## Demo Metrics to Highlight

### Performance
- Query processing: ~2-5 seconds
- Document analysis: ~5-10 seconds
- Multi-API coordination: Seamless

### Accuracy
- WHO ICD medical definitions
- Real-time FDA safety data
- AI-enhanced explanations

### Safety
- 100% response disclaimer coverage
- Emergency detection capability
- Input validation and sanitization

## Post-Demo Q&A Preparation

### Technical Questions
- **"How do you ensure medical accuracy?"** → WHO ICD + FDA APIs
- **"What about liability?"** → Disclaimers + educational purpose
- **"Can it diagnose?"** → No, information only + professional referral

### Business Questions
- **"Who is the target user?"** → Patients, caregivers, health educators
- **"What's the competitive advantage?"** → Multi-API integration + safety focus
- **"How does it scale?"** → Cloud-ready architecture + caching

### Future Roadmap
- Additional medical APIs
- Multilingual support
- Healthcare provider integration
- Mobile application