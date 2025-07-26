# 🏥 Healthcare Navigator Architecture

## System Overview

The Intelligent Healthcare Navigator follows a **ReAct (Reasoning-Acting-Observation)** pattern with a modular, safety-first architecture designed for medical information processing.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           USER INTERFACES                                  │
├─────────────────────┬─────────────────────┬─────────────────────────────────┤
│    CLI Interface    │   Streamlit Web UI  │      Future: API/Slack Bot      │
│                     │                     │                                 │
│ • Interactive CLI   │ • Chat Interface    │ • REST API endpoints            │
│ • Direct commands   │ • File Upload       │ • Webhook integrations          │
│ • Status monitoring │ • User Preferences  │ • Third-party integrations      │
└─────────────────────┴─────────────────────┴─────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            AGENT CORE                                      │
│                        (ReAct Pattern)                                     │
├─────────────────────┬─────────────────────┬─────────────────────────────────┤
│      PLANNER        │      EXECUTOR       │         MEMORY                  │
│                     │                     │                                 │
│ ┌─────────────────┐ │ ┌─────────────────┐ │ ┌─────────────────────────────┐ │
│ │ Query Planner   │ │ │ Tool Executor   │ │ │ Conversation Memory         │ │
│ │                 │ │ │                 │ │ │                             │ │
│ │ • Query Analysis│ │ │ • API Calls     │ │ │ • SQLite Backend            │ │
│ │ • Intent Detect │ │ │ • Tool Coord    │ │ │ • Session Management        │ │
│ │ • Plan Creation │ │ │ • Result Synth  │ │ │ • Context Retrieval         │ │
│ └─────────────────┘ │ └─────────────────┘ │ └─────────────────────────────┘ │
│                     │                     │                                 │
│ ┌─────────────────┐ │ ┌─────────────────┐ │ ┌─────────────────────────────┐ │
│ │Context Planner  │ │ │ Medical NLP     │ │ │ Cache Manager               │ │
│ │                 │ │ │                 │ │ │                             │ │
│ │ • History Aware │ │ │ • Entity Extract│ │ │ • API Response Cache        │ │
│ │ • Context Merge │ │ │ • Doc Summarize │ │ │ • TTL Management            │ │
│ │ • Plan Optimize │ │ │ • Text Process  │ │ │ • Memory Optimization       │ │
│ └─────────────────┘ │ └─────────────────┘ │ └─────────────────────────────┘ │
│                     │                     │                                 │
│                     │ ┌─────────────────┐ │ ┌─────────────────────────────┐ │
│                     │ │ Document Proc   │ │ │ User Preferences            │ │
│                     │ │                 │ │ │                             │ │
│                     │ │ • PDF/DOC Parse │ │ │ • Age, Allergies            │ │
│                     │ │ • Text Extract  │ │ │ • Medical History           │ │
│                     │ │ • File Validate │ │ │ • Preference Persistence    │ │
│                     │ └─────────────────┘ │ └─────────────────────────────┘ │
└─────────────────────┴─────────────────────┴─────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TOOL INTEGRATIONS                                   │
├─────────────────────┬─────────────────────┬─────────────────────────────────┤
│   REASONING ENGINE  │    MEDICAL APIs     │      NLP PROCESSING             │
│                     │                     │                                 │
│ ┌─────────────────┐ │ ┌─────────────────┐ │ ┌─────────────────────────────┐ │
│ │ Google Gemini   │ │ │ WHO ICD-11 API  │ │ │ MedSpaCy                    │ │
│ │                 │ │ │                 │ │ │                             │ │
│ │ • gemini-2.0    │ │ │ • OAuth2 Auth   │ │ │ • Medical NER               │ │
│ │ • Function Call │ │ │ • Disease Codes │ │ │ • Entity Recognition        │ │
│ │ • Reasoning     │ │ │ • Definitions   │ │ │ • Medical Text Processing   │ │
│ │ • Text Gen      │ │ │ • Fallback      │ │ │ • Confidence Scoring        │ │
│ └─────────────────┘ │ └─────────────────┘ │ └─────────────────────────────┘ │
│                     │                     │                                 │
│                     │ ┌─────────────────┐ │ ┌─────────────────────────────┐ │
│                     │ │ OpenFDA API     │ │ │ Document Summarization      │ │
│                     │ │                 │ │ │                             │ │
│                     │ │ • Drug Recalls  │ │ │ • SummerTime Library        │ │
│                     │ │ • Adverse Events│ │ │ • BART/T5 Models            │ │
│                     │ │ • Safety Data   │ │ │ • Medical Summarization     │ │
│                     │ │ • Rate Limiting │ │ │ • Key Finding Extraction    │ │
│                     │ └─────────────────┘ │ └─────────────────────────────┘ │
└─────────────────────┴─────────────────────┴─────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    OBSERVABILITY & SAFETY                                  │
├─────────────────────┬─────────────────────┬─────────────────────────────────┤
│      LOGGING        │   ERROR HANDLING    │      SAFETY SYSTEMS             │
│                     │                     │                                 │
│ • Structured Logs   │ • Retry Logic       │ • Input Sanitization            │
│ • Performance Trace │ • Graceful Degrade  │ • Medical Disclaimers           │
│ • API Call Tracking │ • Fallback Chains   │ • Emergency Detection           │
│ • User Interactions │ • Circuit Breakers  │ • Allergy Warnings              │
│ • Debug Information │ • Timeout Handling  │ • Professional Consultation     │
└─────────────────────┴─────────────────────┴─────────────────────────────────┘
```

## Core Components

### 1. User Interfaces

#### CLI Interface (`cli.py`)
- **Interactive Mode**: Real-time conversation with the agent
- **Direct Commands**: Single-shot queries with immediate responses
- **File Processing**: Document upload and analysis
- **System Monitoring**: Health checks and status reporting

#### Streamlit Web UI (`web_app.py`)
- **Chat Interface**: Conversational medical information access
- **Document Upload**: Drag-and-drop medical document processing
- **User Preferences**: Age, allergies, medical history management
- **Session Management**: Conversation history and context preservation
- **Response Formatting**: Markdown rendering with medical term highlighting

### 2. Agent Core (ReAct Pattern)

#### Planner Module (`src/planner.py`)
```python
class QueryPlanner:
    def analyze_query(self, query: MedicalQuery) -> QueryPlan
    def determine_tools_needed(self, query_type: QueryType) -> List[str]
    def create_execution_steps(self, analysis: Dict) -> List[ExecutionStep]
```

**Responsibilities:**
- **Query Classification**: Medical term, drug info, symptoms, documents, entities
- **Intent Detection**: Urgency level assessment and priority assignment
- **Tool Selection**: Determine required APIs and processing tools
- **Plan Generation**: Create step-by-step execution strategy

#### Context-Aware Planner (`src/context_planner.py`)
```python
class ContextAwarePlanner:
    def create_context_aware_plan(self, query: MedicalQuery, context: Dict) -> QueryPlan
    def integrate_conversation_history(self, plan: QueryPlan) -> QueryPlan
    def optimize_for_user_preferences(self, plan: QueryPlan) -> QueryPlan
```

**Advanced Features:**
- **Conversation Context**: Integrates previous interactions for continuity
- **User Personalization**: Considers age, allergies, medical history
- **Plan Optimization**: Reduces redundant API calls and improves efficiency

#### Executor Module (`src/executor.py`)
```python
class ToolExecutor:
    def execute_plan(self, plan: QueryPlan) -> MedicalResponse
    def execute_medical_term_lookup(self, input_data: Dict) -> Dict
    def execute_drug_info_lookup(self, input_data: Dict) -> Dict
    def execute_symptom_analysis(self, input_data: Dict) -> Dict
    def execute_document_summary(self, input_data: Dict) -> Dict
    def execute_entity_extraction(self, input_data: Dict) -> Dict
```

**Tool Coordination:**
- **Parallel Execution**: Multiple API calls when beneficial
- **Retry Logic**: Exponential backoff for transient failures
- **Timeout Management**: Prevents hanging operations
- **Result Synthesis**: Combines multiple data sources into coherent responses

### 3. Memory Structure

#### Conversation Memory (`src/memory.py`)
```sql
-- SQLite Schema
CREATE TABLE conversations (
    id INTEGER PRIMARY KEY,
    session_id TEXT,
    query_text TEXT,
    response_text TEXT,
    metadata JSON,
    timestamp DATETIME,
    confidence_score REAL
);

CREATE TABLE user_preferences (
    session_id TEXT,
    key TEXT,
    value TEXT,
    updated_at DATETIME,
    PRIMARY KEY (session_id, key)
);
```

**Features:**
- **Session Management**: Isolated conversations per user
- **Context Retrieval**: Relevant history for query processing
- **Metadata Storage**: Confidence scores, sources, processing times
- **Privacy Controls**: Configurable retention policies

#### Cache Manager
```python
class CacheManager:
    def cache_api_response(self, key: str, response: Dict, ttl: int)
    def get_cached_response(self, key: str) -> Optional[Dict]
    def invalidate_cache(self, pattern: str)
    def get_cache_stats(self) -> Dict
```

**Caching Strategy:**
- **API Response Caching**: Reduces external API calls
- **TTL Management**: Configurable expiration times
- **Memory Optimization**: LRU eviction for memory management
- **Cache Warming**: Pre-populate common medical terms

## Tool Integrations

### Google Gemini API (`src/gemini_client.py`)

**Configuration:**
- **Model**: `gemini-2.0-flash` for optimal performance
- **Function Calling**: Structured tool integration
- **Safety Settings**: Medical-appropriate content filtering
- **Rate Limiting**: Intelligent request throttling

**Healthcare-Specific Features:**
```python
def _get_healthcare_system_instructions(self) -> str:
    return """
    You are an intelligent healthcare navigator AI assistant.
    
    CRITICAL GUIDELINES:
    - You do NOT provide medical diagnoses
    - Always include appropriate medical disclaimers
    - Recommend consulting healthcare professionals
    - For urgent symptoms, advise seeking immediate medical attention
    
    RESPONSE REQUIREMENTS:
    - Provide ONLY the final, user-friendly response
    - Do NOT show reasoning process or function calls
    - Use plain language, avoiding unnecessary medical jargon
    - Structure information logically with clear sections
    """
```

### WHO ICD-11 API (`src/who_icd_client.py`)

**Authentication:**
- **OAuth2 Client Credentials**: Secure API access
- **Token Management**: Automatic refresh and caching
- **Error Handling**: Graceful fallback for authentication failures

**Medical Term Processing:**
```python
class WHOICDClient:
    def search_medical_term(self, term: str) -> APIResult
    def get_disease_definition(self, icd_code: str) -> Dict
    def search_by_category(self, category: str) -> List[Dict]
```

### OpenFDA API (`src/openfda_client.py`)

**Drug Safety Information:**
- **Recall Data**: Recent drug recalls and safety alerts
- **Adverse Events**: Reported side effects and reactions
- **Label Information**: Official drug labeling data
- **Enforcement Reports**: FDA enforcement actions

**Rate Limiting & Caching:**
```python
class OpenFDAClient:
    def __init__(self):
        self.rate_limiter = RateLimiter(requests_per_minute=240)
        self.cache = TTLCache(maxsize=1000, ttl=3600)
    
    async def get_drug_information(self, drug_name: str) -> APIResult
    async def search_drug_recalls(self, drug_name: str) -> List[Dict]
    async def get_adverse_events(self, drug_name: str) -> Dict
```

### Medical NLP Processing (`src/medical_nlp.py`)

**Entity Extraction:**
- **MedSpaCy**: Medical named entity recognition
- **Drug NER**: Specialized drug name extraction
- **Confidence Scoring**: Entity recognition confidence levels
- **Type Classification**: Disease, drug, symptom, procedure categorization

**Document Summarization:**
- **SummerTime Library**: Multiple summarization models
- **Medical Context**: Healthcare-specific summarization
- **Key Finding Extraction**: Important medical information highlighting
- **Structured Output**: Organized summary with sections

## Logging and Observability

### Structured Logging (`src/utils.py`)

```python
def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/healthcare_navigator_{datetime.now().strftime("%Y%m%d")}.log'),
            logging.StreamHandler()
        ]
    )
```

**Log Categories:**
- **User Interactions**: Query processing, response generation
- **API Calls**: External service requests and responses
- **Performance Metrics**: Processing times, cache hit rates
- **Error Tracking**: Failures, retries, fallback activations
- **Security Events**: Input validation, sanitization actions

### Performance Monitoring

**Metrics Tracked:**
```python
@dataclass
class PerformanceMetrics:
    query_processing_time: float
    api_response_times: Dict[str, float]
    cache_hit_rate: float
    error_rate: float
    user_satisfaction_score: float
```

**Health Checks:**
```python
async def health_check() -> Dict[str, Any]:
    return {
        'system_healthy': bool,
        'api_status': {
            'gemini': 'healthy|degraded|down',
            'who_icd': 'healthy|degraded|down',
            'openfda': 'healthy|degraded|down'
        },
        'memory_usage': memory_stats,
        'cache_stats': cache_metrics,
        'uptime': system_uptime
    }
```

### Error Handling & Recovery

**Fallback Chain:**
1. **Primary API** → 2. **Secondary API** → 3. **Cached Response** → 4. **AI Fallback** → 5. **Error Message**

**Circuit Breaker Pattern:**
```python
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
```

**Retry Strategy:**
- **Exponential Backoff**: Increasing delays between retries
- **Jitter**: Random delay variation to prevent thundering herd
- **Max Attempts**: Configurable retry limits per operation
- **Timeout Handling**: Per-operation timeout management

## Security & Safety Architecture

### Input Validation Pipeline
```python
def sanitize_input(user_input: str) -> str:
    # 1. HTML/JavaScript injection prevention
    # 2. SQL injection pattern detection
    # 3. Command injection protection
    # 4. Medical query validation
    # 5. Length and format validation
```

### Medical Safety Systems
- **Disclaimer Injection**: Automatic medical disclaimer inclusion
- **Emergency Detection**: Urgent symptom pattern recognition
- **Professional Referral**: Healthcare provider consultation reminders
- **Allergy Checking**: User allergy cross-reference system

### Data Privacy Controls
- **Session Isolation**: User data separation
- **Temporary Storage**: No persistent medical query storage
- **Configurable Retention**: Adjustable data retention policies
- **Audit Logging**: Complete interaction audit trail

## Deployment Architecture

### Development Environment
```bash
# Local development stack
├── Python 3.8+ runtime
├── SQLite database (development)
├── File-based logging
├── In-memory caching
└── Environment-based configuration
```

### Production Environment
```bash
# Production deployment options
├── Docker containerization
├── PostgreSQL database (production)
├── Centralized logging (ELK stack)
├── Redis caching cluster
├── Load balancing (nginx)
├── SSL/TLS termination
└── Environment secret management
```

### Scalability Considerations
- **Horizontal Scaling**: Stateless agent design
- **Database Sharding**: Session-based data partitioning
- **API Rate Limiting**: Distributed rate limiting
- **Caching Strategy**: Multi-tier caching architecture
- **Load Balancing**: Request distribution across instances

This architecture ensures the Healthcare Navigator provides reliable, safe, and scalable medical information access while maintaining the highest standards of medical safety and user privacy.