"""
Medical NLP and entity extraction for Intelligent Healthcare Navigator
Integrates with Fast Data Science libraries and medical NER tools
"""

import re
from typing import List, Dict, Any, Optional, Tuple, Set
import json
from dataclasses import dataclass
from enum import Enum
import asyncio

from src.models import MedicalEntity, EntityType
from src.utils import setup_logging

logger = setup_logging()

class ConfidenceLevel(Enum):
    """Confidence levels for entity extraction"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class ExtractionResult:
    """Result container for entity extraction"""
    entities: List[MedicalEntity]
    confidence_distribution: Dict[str, int]
    processing_time: float
    method_used: str
    text_length: int

class MedicalEntityExtractor:
    """Medical entity extraction using multiple NLP approaches"""
    
    def __init__(self):
        """Initialize medical entity extractor"""
        self.drug_patterns = self._load_drug_patterns()
        self.disease_patterns = self._load_disease_patterns()
        self.symptom_patterns = self._load_symptom_patterns()
        self.procedure_patterns = self._load_procedure_patterns()
        
        # Medical abbreviations and their expansions
        self.medical_abbreviations = self._load_medical_abbreviations()
        
        # Drug name variations and common misspellings
        self.drug_variations = self._load_drug_variations()
        
        logger.info("Medical entity extractor initialized")
    
    def _load_drug_patterns(self) -> Dict[str, List[str]]:
        """Load drug name patterns and common medications"""
        return {
            'common_drugs': [
                'aspirin', 'ibuprofen', 'acetaminophen', 'paracetamol', 'tylenol',
                'advil', 'motrin', 'aleve', 'naproxen', 'diclofenac',
                'metformin', 'insulin', 'lisinopril', 'amlodipine', 'atorvastatin',
                'simvastatin', 'omeprazole', 'pantoprazole', 'levothyroxine',
                'warfarin', 'clopidogrel', 'metoprolol', 'losartan', 'furosemide',
                'hydrochlorothiazide', 'prednisone', 'prednisolone', 'amoxicillin',
                'azithromycin', 'ciprofloxacin', 'doxycycline', 'cephalexin'
            ],
            'drug_suffixes': [
                'cillin', 'mycin', 'floxacin', 'cycline', 'prazole', 'sartan',
                'pril', 'olol', 'pine', 'statin', 'ide', 'ine', 'one'
            ],
            'drug_prefixes': [
                'anti', 'pro', 'pre', 'post', 'meta', 'hydro', 'chloro',
                'fluoro', 'nitro', 'sulfa', 'ceph', 'amox', 'azith'
            ]
        }
    
    def _load_disease_patterns(self) -> Dict[str, List[str]]:
        """Load disease and condition patterns"""
        return {
            'common_diseases': [
                'diabetes', 'hypertension', 'hypotension', 'asthma', 'copd',
                'pneumonia', 'bronchitis', 'influenza', 'covid', 'coronavirus',
                'cancer', 'tumor', 'carcinoma', 'lymphoma', 'leukemia',
                'arthritis', 'osteoarthritis', 'rheumatoid arthritis',
                'heart disease', 'coronary artery disease', 'myocardial infarction',
                'stroke', 'seizure', 'epilepsy', 'migraine', 'depression',
                'anxiety', 'bipolar', 'schizophrenia', 'alzheimer', 'dementia',
                'parkinson', 'multiple sclerosis', 'lupus', 'fibromyalgia',
                'osteoporosis', 'anemia', 'thyroid', 'hypothyroidism', 'hyperthyroidism'
            ],
            'disease_suffixes': [
                'itis', 'osis', 'emia', 'uria', 'pathy', 'trophy', 'plasia',
                'carcinoma', 'sarcoma', 'lymphoma', 'leukemia', 'syndrome'
            ],
            'condition_indicators': [
                'disease', 'disorder', 'condition', 'syndrome', 'infection',
                'inflammation', 'deficiency', 'insufficiency', 'failure'
            ]
        }
    
    def _load_symptom_patterns(self) -> Dict[str, List[str]]:
        """Load symptom patterns"""
        return {
            'common_symptoms': [
                'pain', 'ache', 'headache', 'backache', 'stomachache',
                'fever', 'chills', 'sweating', 'fatigue', 'weakness',
                'nausea', 'vomiting', 'diarrhea', 'constipation',
                'cough', 'shortness of breath', 'wheezing', 'chest pain',
                'dizziness', 'lightheadedness', 'fainting', 'confusion',
                'rash', 'itching', 'swelling', 'bruising', 'bleeding',
                'numbness', 'tingling', 'burning', 'stiffness',
                'insomnia', 'drowsiness', 'anxiety', 'depression',
                'loss of appetite', 'weight loss', 'weight gain',
                'frequent urination', 'difficulty urinating', 'blood in urine',
                'blurred vision', 'double vision', 'hearing loss', 'tinnitus'
            ],
            'pain_descriptors': [
                'sharp', 'dull', 'throbbing', 'burning', 'stabbing',
                'cramping', 'aching', 'shooting', 'radiating'
            ],
            'severity_indicators': [
                'mild', 'moderate', 'severe', 'excruciating', 'unbearable',
                'slight', 'intense', 'chronic', 'acute', 'persistent'
            ]
        }
    
    def _load_procedure_patterns(self) -> Dict[str, List[str]]:
        """Load medical procedure patterns"""
        return {
            'common_procedures': [
                'surgery', 'operation', 'biopsy', 'endoscopy', 'colonoscopy',
                'x-ray', 'ct scan', 'mri', 'ultrasound', 'ecg', 'ekg',
                'blood test', 'urine test', 'stool test', 'culture',
                'vaccination', 'immunization', 'injection', 'infusion',
                'dialysis', 'chemotherapy', 'radiation therapy',
                'physical therapy', 'occupational therapy', 'speech therapy',
                'anesthesia', 'intubation', 'catheterization',
                'transplant', 'bypass', 'stent', 'pacemaker'
            ],
            'procedure_suffixes': [
                'scopy', 'graphy', 'tomy', 'ectomy', 'plasty', 'centesis',
                'ostomy', 'therapy', 'analysis', 'screening'
            ],
            'diagnostic_procedures': [
                'diagnosis', 'examination', 'assessment', 'evaluation',
                'screening', 'monitoring', 'follow-up', 'consultation'
            ]
        }
    
    def _load_medical_abbreviations(self) -> Dict[str, str]:
        """Load common medical abbreviations"""
        return {
            'bp': 'blood pressure',
            'hr': 'heart rate',
            'rr': 'respiratory rate',
            'temp': 'temperature',
            'wbc': 'white blood cell',
            'rbc': 'red blood cell',
            'hgb': 'hemoglobin',
            'hct': 'hematocrit',
            'bun': 'blood urea nitrogen',
            'cr': 'creatinine',
            'na': 'sodium',
            'k': 'potassium',
            'cl': 'chloride',
            'co2': 'carbon dioxide',
            'glucose': 'blood glucose',
            'a1c': 'hemoglobin a1c',
            'tsh': 'thyroid stimulating hormone',
            't3': 'triiodothyronine',
            't4': 'thyroxine',
            'ldl': 'low density lipoprotein',
            'hdl': 'high density lipoprotein',
            'tg': 'triglycerides',
            'pt': 'prothrombin time',
            'ptt': 'partial thromboplastin time',
            'inr': 'international normalized ratio',
            'esr': 'erythrocyte sedimentation rate',
            'crp': 'c-reactive protein'
        }
    
    def _load_drug_variations(self) -> Dict[str, List[str]]:
        """Load drug name variations and common misspellings"""
        return {
            'acetaminophen': ['tylenol', 'paracetamol', 'apap'],
            'ibuprofen': ['advil', 'motrin', 'nurofen'],
            'aspirin': ['asa', 'acetylsalicylic acid', 'bayer'],
            'naproxen': ['aleve', 'naprosyn'],
            'omeprazole': ['prilosec', 'losec'],
            'pantoprazole': ['protonix'],
            'esomeprazole': ['nexium'],
            'lansoprazole': ['prevacid'],
            'metformin': ['glucophage', 'fortamet'],
            'atorvastatin': ['lipitor'],
            'simvastatin': ['zocor'],
            'rosuvastatin': ['crestor'],
            'lisinopril': ['prinivil', 'zestril'],
            'amlodipine': ['norvasc'],
            'metoprolol': ['lopressor', 'toprol'],
            'warfarin': ['coumadin', 'jantoven']
        }
    
    async def extract_entities(
        self, 
        text: str, 
        entity_types: Optional[List[str]] = None,
        confidence_threshold: float = 0.5
    ) -> ExtractionResult:
        """Extract medical entities from text using multiple methods"""
        import time
        start_time = time.time()
        
        logger.info(f"Extracting entities from text ({len(text)} chars)")
        
        # Normalize text
        normalized_text = self._normalize_text(text)
        
        # Extract entities using different methods
        entities = []
        
        # Method 1: Pattern-based extraction
        pattern_entities = await self._extract_with_patterns(normalized_text, entity_types)
        entities.extend(pattern_entities)
        
        # Method 2: Rule-based extraction
        rule_entities = await self._extract_with_rules(normalized_text, entity_types)
        entities.extend(rule_entities)
        
        # Method 3: Context-based extraction
        context_entities = await self._extract_with_context(normalized_text, entity_types)
        entities.extend(context_entities)
        
        # Remove duplicates and filter by confidence
        unique_entities = self._deduplicate_entities(entities)
        filtered_entities = [e for e in unique_entities if e.confidence >= confidence_threshold]
        
        # Calculate confidence distribution
        confidence_dist = self._calculate_confidence_distribution(filtered_entities)
        
        processing_time = time.time() - start_time
        
        logger.info(f"Extracted {len(filtered_entities)} entities in {processing_time:.2f}s")
        
        return ExtractionResult(
            entities=filtered_entities,
            confidence_distribution=confidence_dist,
            processing_time=processing_time,
            method_used="hybrid_pattern_rule_context",
            text_length=len(text)
        )
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for better entity extraction"""
        # Convert to lowercase
        normalized = text.lower()
        
        # Expand common abbreviations
        for abbrev, expansion in self.medical_abbreviations.items():
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            normalized = re.sub(pattern, expansion, normalized)
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    async def _extract_with_patterns(
        self, 
        text: str, 
        entity_types: Optional[List[str]] = None
    ) -> List[MedicalEntity]:
        """Extract entities using pattern matching"""
        entities = []
        
        target_types = entity_types or ['disease', 'drug', 'symptom', 'procedure']
        
        if 'drug' in target_types:
            entities.extend(self._extract_drugs_by_pattern(text))
        
        if 'disease' in target_types:
            entities.extend(self._extract_diseases_by_pattern(text))
        
        if 'symptom' in target_types:
            entities.extend(self._extract_symptoms_by_pattern(text))
        
        if 'procedure' in target_types:
            entities.extend(self._extract_procedures_by_pattern(text))
        
        return entities
    
    def _extract_drugs_by_pattern(self, text: str) -> List[MedicalEntity]:
        """Extract drug entities using patterns"""
        entities = []
        
        # Exact matches for common drugs
        for drug in self.drug_patterns['common_drugs']:
            pattern = r'\b' + re.escape(drug) + r'\b'
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                entity = MedicalEntity(
                    text=match.group(),
                    entity_type=EntityType.DRUG,
                    confidence=0.9,
                    source_api="pattern_matching",
                    start_pos=match.start(),
                    end_pos=match.end(),
                    additional_info={'method': 'exact_match'}
                )
                entities.append(entity)
        
        # Check drug variations
        for main_drug, variations in self.drug_variations.items():
            for variation in variations:
                pattern = r'\b' + re.escape(variation) + r'\b'
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    entity = MedicalEntity(
                        text=match.group(),
                        entity_type=EntityType.DRUG,
                        confidence=0.85,
                        source_api="pattern_matching",
                        start_pos=match.start(),
                        end_pos=match.end(),
                        additional_info={
                            'method': 'variation_match',
                            'canonical_name': main_drug
                        }
                    )
                    entities.append(entity)
        
        # Pattern-based detection for drug-like terms
        for suffix in self.drug_patterns['drug_suffixes']:
            pattern = r'\b\w+' + re.escape(suffix) + r'\b'
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                # Skip if already found as exact match
                if not any(e.text.lower() == match.group().lower() for e in entities):
                    entity = MedicalEntity(
                        text=match.group(),
                        entity_type=EntityType.DRUG,
                        confidence=0.7,
                        source_api="pattern_matching",
                        start_pos=match.start(),
                        end_pos=match.end(),
                        additional_info={'method': 'suffix_pattern'}
                    )
                    entities.append(entity)
        
        return entities
    
    def _extract_diseases_by_pattern(self, text: str) -> List[MedicalEntity]:
        """Extract disease entities using patterns"""
        entities = []
        
        # Exact matches for common diseases
        for disease in self.disease_patterns['common_diseases']:
            pattern = r'\b' + re.escape(disease) + r'\b'
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                entity = MedicalEntity(
                    text=match.group(),
                    entity_type=EntityType.DISEASE,
                    confidence=0.9,
                    source_api="pattern_matching",
                    start_pos=match.start(),
                    end_pos=match.end(),
                    additional_info={'method': 'exact_match'}
                )
                entities.append(entity)
        
        # Pattern-based detection for disease-like terms
        for suffix in self.disease_patterns['disease_suffixes']:
            pattern = r'\b\w+' + re.escape(suffix) + r'\b'
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                if not any(e.text.lower() == match.group().lower() for e in entities):
                    entity = MedicalEntity(
                        text=match.group(),
                        entity_type=EntityType.DISEASE,
                        confidence=0.75,
                        source_api="pattern_matching",
                        start_pos=match.start(),
                        end_pos=match.end(),
                        additional_info={'method': 'suffix_pattern'}
                    )
                    entities.append(entity)
        
        return entities
    
    def _extract_symptoms_by_pattern(self, text: str) -> List[MedicalEntity]:
        """Extract symptom entities using patterns"""
        entities = []
        
        # Exact matches for common symptoms
        for symptom in self.symptom_patterns['common_symptoms']:
            pattern = r'\b' + re.escape(symptom) + r'\b'
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                entity = MedicalEntity(
                    text=match.group(),
                    entity_type=EntityType.SYMPTOM,
                    confidence=0.85,
                    source_api="pattern_matching",
                    start_pos=match.start(),
                    end_pos=match.end(),
                    additional_info={'method': 'exact_match'}
                )
                entities.append(entity)
        
        # Enhanced pain detection with descriptors
        pain_pattern = r'\b(?:' + '|'.join(self.symptom_patterns['pain_descriptors']) + r')\s+pain\b'
        matches = re.finditer(pain_pattern, text, re.IGNORECASE)
        
        for match in matches:
            entity = MedicalEntity(
                text=match.group(),
                entity_type=EntityType.SYMPTOM,
                confidence=0.9,
                source_api="pattern_matching",
                start_pos=match.start(),
                end_pos=match.end(),
                additional_info={'method': 'enhanced_pain_detection'}
            )
            entities.append(entity)
        
        return entities
    
    def _extract_procedures_by_pattern(self, text: str) -> List[MedicalEntity]:
        """Extract procedure entities using patterns"""
        entities = []
        
        # Exact matches for common procedures
        for procedure in self.procedure_patterns['common_procedures']:
            pattern = r'\b' + re.escape(procedure) + r'\b'
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                entity = MedicalEntity(
                    text=match.group(),
                    entity_type=EntityType.PROCEDURE,
                    confidence=0.8,
                    source_api="pattern_matching",
                    start_pos=match.start(),
                    end_pos=match.end(),
                    additional_info={'method': 'exact_match'}
                )
                entities.append(entity)
        
        return entities
    
    async def _extract_with_rules(
        self, 
        text: str, 
        entity_types: Optional[List[str]] = None
    ) -> List[MedicalEntity]:
        """Extract entities using rule-based methods"""
        entities = []
        
        # Rule 1: Dosage patterns indicate drugs
        dosage_pattern = r'\b(\w+)\s+(?:\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|cc|units?|iu)\b)'
        matches = re.finditer(dosage_pattern, text, re.IGNORECASE)
        
        for match in matches:
            drug_name = match.group(1)
            if len(drug_name) > 2:  # Avoid single letters
                entity = MedicalEntity(
                    text=drug_name,
                    entity_type=EntityType.DRUG,
                    confidence=0.8,
                    source_api="rule_based",
                    start_pos=match.start(1),
                    end_pos=match.end(1),
                    additional_info={
                        'method': 'dosage_rule',
                        'full_match': match.group()
                    }
                )
                entities.append(entity)
        
        # Rule 2: "diagnosed with" patterns indicate diseases
        diagnosis_pattern = r'(?:diagnosed with|diagnosis of|suffering from|has)\s+([a-zA-Z\s]+?)(?:\.|,|;|$)'
        matches = re.finditer(diagnosis_pattern, text, re.IGNORECASE)
        
        for match in matches:
            condition = match.group(1).strip()
            if len(condition) > 3 and len(condition.split()) <= 4:  # Reasonable length
                entity = MedicalEntity(
                    text=condition,
                    entity_type=EntityType.DISEASE,
                    confidence=0.85,
                    source_api="rule_based",
                    start_pos=match.start(1),
                    end_pos=match.end(1),
                    additional_info={'method': 'diagnosis_rule'}
                )
                entities.append(entity)
        
        # Rule 3: "experiencing" or "feeling" patterns indicate symptoms
        symptom_pattern = r'(?:experiencing|feeling|having|complains? of)\s+([a-zA-Z\s]+?)(?:\.|,|;|and|$)'
        matches = re.finditer(symptom_pattern, text, re.IGNORECASE)
        
        for match in matches:
            symptom = match.group(1).strip()
            if len(symptom) > 3 and len(symptom.split()) <= 3:
                entity = MedicalEntity(
                    text=symptom,
                    entity_type=EntityType.SYMPTOM,
                    confidence=0.75,
                    source_api="rule_based",
                    start_pos=match.start(1),
                    end_pos=match.end(1),
                    additional_info={'method': 'symptom_rule'}
                )
                entities.append(entity)
        
        return entities
    
    async def _extract_with_context(
        self, 
        text: str, 
        entity_types: Optional[List[str]] = None
    ) -> List[MedicalEntity]:
        """Extract entities using contextual analysis"""
        entities = []
        
        # Context 1: Medical history sections
        history_pattern = r'(?:medical history|past medical history|history of)[\s:]+([^.]+)'
        matches = re.finditer(history_pattern, text, re.IGNORECASE)
        
        for match in matches:
            history_text = match.group(1)
            # Extract potential conditions from history
            conditions = re.findall(r'\b[a-zA-Z]{4,}\b', history_text)
            
            for condition in conditions:
                if condition.lower() in [d.lower() for d in self.disease_patterns['common_diseases']]:
                    entity = MedicalEntity(
                        text=condition,
                        entity_type=EntityType.DISEASE,
                        confidence=0.7,
                        source_api="context_based",
                        additional_info={'method': 'medical_history_context'}
                    )
                    entities.append(entity)
        
        # Context 2: Medication lists
        medication_pattern = r'(?:medications?|drugs?|taking|prescribed)[\s:]+([^.]+)'
        matches = re.finditer(medication_pattern, text, re.IGNORECASE)
        
        for match in matches:
            med_text = match.group(1)
            # Extract potential drugs from medication context
            drugs = re.findall(r'\b[a-zA-Z]{4,}\b', med_text)
            
            for drug in drugs:
                if drug.lower() in [d.lower() for d in self.drug_patterns['common_drugs']]:
                    entity = MedicalEntity(
                        text=drug,
                        entity_type=EntityType.DRUG,
                        confidence=0.8,
                        source_api="context_based",
                        additional_info={'method': 'medication_context'}
                    )
                    entities.append(entity)
        
        return entities
    
    def _deduplicate_entities(self, entities: List[MedicalEntity]) -> List[MedicalEntity]:
        """Remove duplicate entities, keeping the highest confidence"""
        entity_map = {}
        
        for entity in entities:
            key = (entity.text.lower(), entity.entity_type.value)
            
            if key not in entity_map or entity.confidence > entity_map[key].confidence:
                entity_map[key] = entity
        
        return list(entity_map.values())
    
    def _calculate_confidence_distribution(self, entities: List[MedicalEntity]) -> Dict[str, int]:
        """Calculate distribution of confidence levels"""
        distribution = {level.value: 0 for level in ConfidenceLevel}
        
        for entity in entities:
            if entity.confidence >= 0.9:
                distribution[ConfidenceLevel.VERY_HIGH.value] += 1
            elif entity.confidence >= 0.8:
                distribution[ConfidenceLevel.HIGH.value] += 1
            elif entity.confidence >= 0.6:
                distribution[ConfidenceLevel.MEDIUM.value] += 1
            else:
                distribution[ConfidenceLevel.LOW.value] += 1
        
        return distribution
    
    def get_entity_statistics(self, entities: List[MedicalEntity]) -> Dict[str, Any]:
        """Get statistics about extracted entities"""
        if not entities:
            return {'total': 0}
        
        type_counts = {}
        confidence_sum = 0
        source_counts = {}
        
        for entity in entities:
            # Count by type
            entity_type = entity.entity_type.value
            type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
            
            # Sum confidence
            confidence_sum += entity.confidence
            
            # Count by source
            source = entity.source_api
            source_counts[source] = source_counts.get(source, 0) + 1
        
        return {
            'total': len(entities),
            'by_type': type_counts,
            'by_source': source_counts,
            'average_confidence': round(confidence_sum / len(entities), 3),
            'highest_confidence': max(e.confidence for e in entities),
            'lowest_confidence': min(e.confidence for e in entities)
        }
    
    async def validate_entities(
        self, 
        entities: List[MedicalEntity],
        external_validator: Optional[callable] = None
    ) -> List[MedicalEntity]:
        """Validate extracted entities using external sources"""
        validated_entities = []
        
        for entity in entities:
            # Basic validation
            if len(entity.text) < 2 or len(entity.text) > 100:
                continue
            
            # Skip common non-medical words
            if entity.text.lower() in ['the', 'and', 'or', 'but', 'with', 'for', 'to', 'of']:
                continue
            
            # External validation if provided
            if external_validator:
                try:
                    is_valid = await external_validator(entity)
                    if not is_valid:
                        continue
                except Exception as e:
                    logger.warning(f"External validation failed for {entity.text}: {e}")
            
            validated_entities.append(entity)
        
        return validated_entities

class DocumentSummarizer:
    """Medical document summarization using multiple approaches"""
    
    def __init__(self, entity_extractor: MedicalEntityExtractor = None):
        """Initialize document summarizer"""
        self.entity_extractor = entity_extractor or MedicalEntityExtractor()
        
        # Document type patterns
        self.document_patterns = self._load_document_patterns()
        
        # Key section identifiers
        self.section_patterns = self._load_section_patterns()
        
        logger.info("Document summarizer initialized")
    
    def _load_document_patterns(self) -> Dict[str, List[str]]:
        """Load patterns to identify document types"""
        return {
            'prescription': [
                'rx', 'prescription', 'prescribed', 'take as directed',
                'sig:', 'dispense', 'refills', 'pharmacy', 'medication list'
            ],
            'lab_report': [
                'lab results', 'laboratory', 'test results', 'reference range',
                'normal', 'abnormal', 'high', 'low', 'specimen', 'collected'
            ],
            'medical_note': [
                'chief complaint', 'history of present illness', 'physical exam',
                'assessment', 'plan', 'diagnosis', 'patient presents', 'subjective'
            ],
            'discharge_summary': [
                'discharge', 'admission', 'hospital course', 'discharge medications',
                'follow-up', 'discharge instructions', 'condition on discharge'
            ],
            'radiology_report': [
                'impression', 'findings', 'technique', 'comparison', 'x-ray',
                'ct', 'mri', 'ultrasound', 'radiologist', 'imaging'
            ]
        }
    
    def _load_section_patterns(self) -> Dict[str, str]:
        """Load patterns for identifying key sections"""
        return {
            'chief_complaint': r'(?:chief complaint|cc)[\s:]+([^\n]+)',
            'history_present_illness': r'(?:history of present illness|hpi)[\s:]+([^.]+(?:\.[^.]+)*)',
            'past_medical_history': r'(?:past medical history|pmh|medical history)[\s:]+([^.]+(?:\.[^.]+)*)',
            'medications': r'(?:medications?|current medications?|meds?)[\s:]+([^.]+(?:\.[^.]+)*)',
            'allergies': r'(?:allergies?|drug allergies?)[\s:]+([^.]+)',
            'physical_exam': r'(?:physical exam|pe|examination)[\s:]+([^.]+(?:\.[^.]+)*)',
            'assessment': r'(?:assessment|impression|diagnosis)[\s:]+([^.]+(?:\.[^.]+)*)',
            'plan': r'(?:plan|treatment plan)[\s:]+([^.]+(?:\.[^.]+)*)',
            'vital_signs': r'(?:vital signs?|vitals?)[\s:]+([^.]+)',
            'lab_results': r'(?:lab results?|laboratory)[\s:]+([^.]+(?:\.[^.]+)*)',
            'discharge_instructions': r'(?:discharge instructions?|instructions?)[\s:]+([^.]+(?:\.[^.]+)*)'
        }
    
    async def summarize_document(
        self, 
        document_text: str, 
        document_type: str = "unknown",
        summary_length: str = "detailed",
        extract_entities: bool = True
    ) -> Dict[str, Any]:
        """Summarize medical document with key information extraction"""
        import time
        start_time = time.time()
        
        logger.info(f"Summarizing {document_type} document ({len(document_text)} chars)")
        
        # Detect document type if unknown
        if document_type == "unknown":
            document_type = self._detect_document_type(document_text)
        
        # Extract key sections
        sections = self._extract_sections(document_text)
        
        # Generate summary based on document type and length
        summary = await self._generate_summary(document_text, document_type, summary_length, sections)
        
        # Extract key findings
        key_findings = self._extract_key_findings(document_text, document_type, sections)
        
        # Extract medical entities if requested
        entities = []
        if extract_entities:
            extraction_result = await self.entity_extractor.extract_entities(document_text)
            entities = extraction_result.entities
        
        # Calculate confidence score
        confidence_score = self._calculate_summary_confidence(
            document_text, summary, sections, document_type
        )
        
        processing_time = time.time() - start_time
        
        result = {
            'document_type': document_type,
            'original_length': len(document_text),
            'summary': summary,
            'key_findings': key_findings,
            'extracted_entities': [entity.to_dict() for entity in entities],
            'sections_found': list(sections.keys()),
            'confidence_score': confidence_score,
            'processing_time': processing_time,
            'summary_length': summary_length,
            'metadata': {
                'word_count_original': len(document_text.split()),
                'word_count_summary': len(summary.split()),
                'compression_ratio': round(len(summary) / len(document_text), 3) if document_text else 0
            }
        }
        
        logger.info(f"Document summarized in {processing_time:.2f}s (confidence: {confidence_score:.2f})")
        
        return result
    
    def _detect_document_type(self, text: str) -> str:
        """Detect document type based on content patterns"""
        text_lower = text.lower()
        type_scores = {}
        
        for doc_type, patterns in self.document_patterns.items():
            score = 0
            for pattern in patterns:
                if pattern in text_lower:
                    score += 1
            
            # Normalize score by number of patterns
            type_scores[doc_type] = score / len(patterns)
        
        # Return type with highest score, or 'unknown' if no clear match
        if type_scores:
            best_type = max(type_scores, key=type_scores.get)
            if type_scores[best_type] > 0.2:  # Threshold for confidence
                return best_type
        
        return 'unknown'
    
    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Extract key sections from medical document"""
        sections = {}
        
        for section_name, pattern in self.section_patterns.items():
            matches = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if matches:
                content = matches.group(1).strip()
                # Clean up content
                content = re.sub(r'\s+', ' ', content)
                sections[section_name] = content
        
        return sections
    
    async def _generate_summary(
        self, 
        text: str, 
        document_type: str, 
        length: str, 
        sections: Dict[str, str]
    ) -> str:
        """Generate summary based on document type and desired length"""
        
        if document_type == 'prescription':
            return self._summarize_prescription(text, sections, length)
        elif document_type == 'lab_report':
            return self._summarize_lab_report(text, sections, length)
        elif document_type == 'medical_note':
            return self._summarize_medical_note(text, sections, length)
        elif document_type == 'discharge_summary':
            return self._summarize_discharge_summary(text, sections, length)
        elif document_type == 'radiology_report':
            return self._summarize_radiology_report(text, sections, length)
        else:
            return self._summarize_generic_document(text, sections, length)
    
    def _summarize_prescription(self, text: str, sections: Dict[str, str], length: str) -> str:
        """Summarize prescription document"""
        summary_parts = []
        
        # Extract medication information
        medications = self._extract_medications_from_text(text)
        
        if medications:
            summary_parts.append(f"Prescribed medications: {', '.join(medications[:5])}")
        
        # Add patient instructions if found
        if 'instructions' in sections:
            instructions = sections['instructions'][:200] if length == 'brief' else sections['instructions']
            summary_parts.append(f"Instructions: {instructions}")
        
        # Add allergies if mentioned
        if 'allergies' in sections:
            summary_parts.append(f"Allergies: {sections['allergies']}")
        
        return '. '.join(summary_parts) if summary_parts else "Prescription document with medication information."
    
    def _summarize_lab_report(self, text: str, sections: Dict[str, str], length: str) -> str:
        """Summarize laboratory report"""
        summary_parts = []
        
        # Extract abnormal values
        abnormal_results = self._extract_abnormal_lab_values(text)
        
        if abnormal_results:
            summary_parts.append(f"Abnormal results: {', '.join(abnormal_results[:3])}")
        
        # Add key lab results
        if 'lab_results' in sections:
            results = sections['lab_results']
            if length == 'brief':
                results = results[:150] + "..." if len(results) > 150 else results
            summary_parts.append(f"Lab findings: {results}")
        
        return '. '.join(summary_parts) if summary_parts else "Laboratory report with test results."
    
    def _summarize_medical_note(self, text: str, sections: Dict[str, str], length: str) -> str:
        """Summarize medical note/clinical note"""
        summary_parts = []
        
        # Chief complaint
        if 'chief_complaint' in sections:
            summary_parts.append(f"Chief complaint: {sections['chief_complaint']}")
        
        # Assessment/Diagnosis
        if 'assessment' in sections:
            assessment = sections['assessment']
            if length == 'brief':
                assessment = assessment[:100] + "..." if len(assessment) > 100 else assessment
            summary_parts.append(f"Assessment: {assessment}")
        
        # Plan
        if 'plan' in sections:
            plan = sections['plan']
            if length == 'brief':
                plan = plan[:100] + "..." if len(plan) > 100 else plan
            summary_parts.append(f"Plan: {plan}")
        
        # Add vital signs if present
        if 'vital_signs' in sections:
            summary_parts.append(f"Vitals: {sections['vital_signs']}")
        
        return '. '.join(summary_parts) if summary_parts else "Medical note with patient encounter information."
    
    def _summarize_discharge_summary(self, text: str, sections: Dict[str, str], length: str) -> str:
        """Summarize discharge summary"""
        summary_parts = []
        
        # Reason for admission
        admission_reason = self._extract_admission_reason(text)
        if admission_reason:
            summary_parts.append(f"Admission reason: {admission_reason}")
        
        # Discharge diagnosis
        discharge_diagnosis = self._extract_discharge_diagnosis(text)
        if discharge_diagnosis:
            summary_parts.append(f"Discharge diagnosis: {discharge_diagnosis}")
        
        # Discharge instructions
        if 'discharge_instructions' in sections:
            instructions = sections['discharge_instructions']
            if length == 'brief':
                instructions = instructions[:150] + "..." if len(instructions) > 150 else instructions
            summary_parts.append(f"Instructions: {instructions}")
        
        return '. '.join(summary_parts) if summary_parts else "Discharge summary with hospital stay information."
    
    def _summarize_radiology_report(self, text: str, sections: Dict[str, str], length: str) -> str:
        """Summarize radiology report"""
        summary_parts = []
        
        # Extract imaging type
        imaging_type = self._extract_imaging_type(text)
        if imaging_type:
            summary_parts.append(f"Imaging: {imaging_type}")
        
        # Extract impression/findings
        impression = self._extract_radiology_impression(text)
        if impression:
            if length == 'brief':
                impression = impression[:200] + "..." if len(impression) > 200 else impression
            summary_parts.append(f"Impression: {impression}")
        
        return '. '.join(summary_parts) if summary_parts else "Radiology report with imaging findings."
    
    def _summarize_generic_document(self, text: str, sections: Dict[str, str], length: str) -> str:
        """Summarize generic medical document"""
        # Use extractive summarization for generic documents
        sentences = self._split_into_sentences(text)
        
        if length == 'brief':
            max_sentences = 2
        elif length == 'comprehensive':
            max_sentences = 8
        else:  # detailed
            max_sentences = 5
        
        # Score sentences by medical relevance
        scored_sentences = self._score_sentences_for_relevance(sentences)
        
        # Select top sentences
        top_sentences = sorted(scored_sentences, key=lambda x: x[1], reverse=True)[:max_sentences]
        
        # Sort by original order
        selected_sentences = sorted(top_sentences, key=lambda x: x[2])
        
        summary = '. '.join([s[0] for s in selected_sentences])
        
        return summary if summary else "Medical document summary not available."
    
    def _extract_key_findings(
        self, 
        text: str, 
        document_type: str, 
        sections: Dict[str, str]
    ) -> List[str]:
        """Extract key findings from document"""
        findings = []
        
        # Document-specific key findings
        if document_type == 'prescription':
            medications = self._extract_medications_from_text(text)
            findings.extend([f"Medication: {med}" for med in medications[:5]])
        
        elif document_type == 'lab_report':
            abnormal_values = self._extract_abnormal_lab_values(text)
            findings.extend([f"Abnormal: {val}" for val in abnormal_values[:5]])
        
        elif document_type == 'medical_note':
            if 'assessment' in sections:
                diagnoses = self._extract_diagnoses_from_text(sections['assessment'])
                findings.extend([f"Diagnosis: {dx}" for dx in diagnoses[:3]])
        
        # Generic findings
        critical_terms = self._extract_critical_terms(text)
        findings.extend([f"Critical: {term}" for term in critical_terms[:3]])
        
        return findings[:10]  # Limit to 10 key findings
    
    def _extract_medications_from_text(self, text: str) -> List[str]:
        """Extract medication names from text"""
        medications = []
        
        # Pattern for medication with dosage
        med_pattern = r'\b([A-Za-z]+(?:cillin|mycin|prazole|sartan|pril|olol|pine|statin))\s+\d+\s*(?:mg|mcg|g|ml|units?)\b'
        matches = re.findall(med_pattern, text, re.IGNORECASE)
        medications.extend(matches)
        
        # Common medication names
        common_meds = self.entity_extractor.drug_patterns['common_drugs']
        for med in common_meds:
            if re.search(r'\b' + re.escape(med) + r'\b', text, re.IGNORECASE):
                medications.append(med)
        
        return list(set(medications))  # Remove duplicates
    
    def _extract_abnormal_lab_values(self, text: str) -> List[str]:
        """Extract abnormal laboratory values"""
        abnormal_indicators = ['high', 'low', 'elevated', 'decreased', 'abnormal', 'critical']
        abnormal_results = []
        
        for indicator in abnormal_indicators:
            pattern = rf'\b({indicator})\s+([A-Za-z\s]+?)(?:\s|$|\.)'
            matches = re.findall(pattern, text, re.IGNORECASE)
            
            for match in matches:
                result = f"{match[1].strip()} ({match[0]})"
                if len(result) < 50:  # Reasonable length
                    abnormal_results.append(result)
        
        return abnormal_results[:5]
    
    def _extract_diagnoses_from_text(self, text: str) -> List[str]:
        """Extract diagnoses from assessment text"""
        diagnoses = []
        
        # Pattern for numbered diagnoses
        numbered_pattern = r'\d+\.\s*([^.]+?)(?:\.|$)'
        matches = re.findall(numbered_pattern, text)
        diagnoses.extend([match.strip() for match in matches])
        
        # Common disease patterns
        diseases = self.entity_extractor.disease_patterns['common_diseases']
        for disease in diseases:
            if re.search(r'\b' + re.escape(disease) + r'\b', text, re.IGNORECASE):
                diagnoses.append(disease)
        
        return list(set(diagnoses))[:5]
    
    def _extract_critical_terms(self, text: str) -> List[str]:
        """Extract critical medical terms that should be highlighted"""
        critical_keywords = [
            'emergency', 'urgent', 'critical', 'severe', 'acute', 'chronic',
            'malignant', 'benign', 'positive', 'negative', 'abnormal', 'normal'
        ]
        
        critical_terms = []
        text_lower = text.lower()
        
        for keyword in critical_keywords:
            if keyword in text_lower:
                # Find context around the keyword
                pattern = rf'\b\w*{re.escape(keyword)}\w*\b'
                matches = re.findall(pattern, text, re.IGNORECASE)
                critical_terms.extend(matches)
        
        return list(set(critical_terms))[:5]
    
    def _extract_admission_reason(self, text: str) -> str:
        """Extract reason for hospital admission"""
        patterns = [
            r'(?:admitted for|admission for|reason for admission)[\s:]+([^.]+)',
            r'(?:chief complaint|cc)[\s:]+([^.]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _extract_discharge_diagnosis(self, text: str) -> str:
        """Extract discharge diagnosis"""
        patterns = [
            r'(?:discharge diagnosis|final diagnosis)[\s:]+([^.]+)',
            r'(?:primary diagnosis)[\s:]+([^.]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _extract_imaging_type(self, text: str) -> str:
        """Extract type of imaging study"""
        imaging_types = ['x-ray', 'ct', 'mri', 'ultrasound', 'mammogram', 'pet scan']
        
        for imaging_type in imaging_types:
            if re.search(r'\b' + re.escape(imaging_type) + r'\b', text, re.IGNORECASE):
                return imaging_type
        
        return ""
    
    def _extract_radiology_impression(self, text: str) -> str:
        """Extract radiology impression/findings"""
        patterns = [
            r'(?:impression|conclusion)[\s:]+([^.]+(?:\.[^.]+)*)',
            r'(?:findings)[\s:]+([^.]+(?:\.[^.]+)*)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def _score_sentences_for_relevance(self, sentences: List[str]) -> List[Tuple[str, float, int]]:
        """Score sentences by medical relevance"""
        medical_keywords = [
            'patient', 'diagnosis', 'treatment', 'medication', 'symptom',
            'condition', 'disease', 'therapy', 'procedure', 'test', 'result'
        ]
        
        scored_sentences = []
        
        for i, sentence in enumerate(sentences):
            score = 0
            sentence_lower = sentence.lower()
            
            # Score based on medical keywords
            for keyword in medical_keywords:
                if keyword in sentence_lower:
                    score += 1
            
            # Bonus for sentences with numbers (likely lab values, dosages)
            if re.search(r'\d+', sentence):
                score += 0.5
            
            # Penalty for very short or very long sentences
            word_count = len(sentence.split())
            if word_count < 5 or word_count > 50:
                score -= 1
            
            scored_sentences.append((sentence, score, i))
        
        return scored_sentences
    
    def _calculate_summary_confidence(
        self, 
        original_text: str, 
        summary: str, 
        sections: Dict[str, str], 
        document_type: str
    ) -> float:
        """Calculate confidence score for the summary"""
        confidence = 0.5  # Base confidence
        
        # Boost confidence if document type was detected
        if document_type != 'unknown':
            confidence += 0.2
        
        # Boost confidence if key sections were found
        section_bonus = min(len(sections) * 0.05, 0.2)
        confidence += section_bonus
        
        # Boost confidence if summary has reasonable length
        if 50 <= len(summary) <= 500:
            confidence += 0.1
        
        # Ensure confidence is between 0 and 1
        return min(max(confidence, 0.0), 1.0)