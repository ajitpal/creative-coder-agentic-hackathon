"""
Unit tests for medical NLP and entity extraction
"""

import pytest
import asyncio
from src.medical_nlp import MedicalEntityExtractor, ExtractionResult, ConfidenceLevel
from src.models import MedicalEntity, EntityType

class TestMedicalEntityExtractor:
    """Test cases for MedicalEntityExtractor class"""
    
    @pytest.fixture
    def extractor(self):
        """Create MedicalEntityExtractor instance for testing"""
        return MedicalEntityExtractor()
    
    def test_extractor_initialization(self, extractor):
        """Test extractor initialization"""
        assert extractor.drug_patterns is not None
        assert extractor.disease_patterns is not None
        assert extractor.symptom_patterns is not None
        assert extractor.procedure_patterns is not None
        assert extractor.medical_abbreviations is not None
        assert extractor.drug_variations is not None
        
        # Check that patterns contain expected data
        assert 'aspirin' in extractor.drug_patterns['common_drugs']
        assert 'diabetes' in extractor.disease_patterns['common_diseases']
        assert 'headache' in extractor.symptom_patterns['common_symptoms']
        assert 'surgery' in extractor.procedure_patterns['common_procedures']
    
    def test_normalize_text(self, extractor):
        """Test text normalization"""
        text = "Patient has elevated BP and HR. WBC count is normal."
        normalized = extractor._normalize_text(text)
        
        assert "blood pressure" in normalized
        assert "heart rate" in normalized
        assert "white blood cell" in normalized
        assert normalized.islower()
    
    def test_extract_drugs_by_pattern_exact_match(self, extractor):
        """Test drug extraction with exact matches"""
        text = "Patient is taking aspirin and ibuprofen for pain relief."
        
        entities = extractor._extract_drugs_by_pattern(text)
        
        drug_names = [e.text.lower() for e in entities]
        assert 'aspirin' in drug_names
        assert 'ibuprofen' in drug_names
        
        # Check confidence levels
        for entity in entities:
            assert entity.entity_type == EntityType.DRUG
            assert entity.confidence >= 0.8
            assert entity.source_api == "pattern_matching"
    
    def test_extract_drugs_by_pattern_variations(self, extractor):
        """Test drug extraction with brand name variations"""
        text = "Patient takes Tylenol and Advil daily."
        
        entities = extractor._extract_drugs_by_pattern(text)
        
        drug_names = [e.text.lower() for e in entities]
        assert 'tylenol' in drug_names
        assert 'advil' in drug_names
        
        # Check that canonical names are recorded
        tylenol_entity = next(e for e in entities if e.text.lower() == 'tylenol')
        assert tylenol_entity.additional_info.get('canonical_name') == 'acetaminophen'
    
    def test_extract_drugs_by_pattern_suffix(self, extractor):
        """Test drug extraction using suffix patterns"""
        text = "Patient is prescribed amoxicillin and azithromycin."
        
        entities = extractor._extract_drugs_by_pattern(text)
        
        drug_names = [e.text.lower() for e in entities]
        assert 'amoxicillin' in drug_names
        assert 'azithromycin' in drug_names
    
    def test_extract_diseases_by_pattern(self, extractor):
        """Test disease extraction"""
        text = "Patient has diabetes and hypertension. Also diagnosed with arthritis."
        
        entities = extractor._extract_diseases_by_pattern(text)
        
        disease_names = [e.text.lower() for e in entities]
        assert 'diabetes' in disease_names
        assert 'hypertension' in disease_names
        assert 'arthritis' in disease_names
        
        for entity in entities:
            assert entity.entity_type == EntityType.DISEASE
            assert entity.confidence >= 0.7
    
    def test_extract_diseases_by_pattern_suffix(self, extractor):
        """Test disease extraction using suffix patterns"""
        text = "Patient shows signs of gastritis and bronchitis."
        
        entities = extractor._extract_diseases_by_pattern(text)
        
        disease_names = [e.text.lower() for e in entities]
        assert 'gastritis' in disease_names
        assert 'bronchitis' in disease_names
    
    def test_extract_symptoms_by_pattern(self, extractor):
        """Test symptom extraction"""
        text = "Patient complains of headache, nausea, and fatigue."
        
        entities = extractor._extract_symptoms_by_pattern(text)
        
        symptom_names = [e.text.lower() for e in entities]
        assert 'headache' in symptom_names
        assert 'nausea' in symptom_names
        assert 'fatigue' in symptom_names
        
        for entity in entities:
            assert entity.entity_type == EntityType.SYMPTOM
    
    def test_extract_symptoms_enhanced_pain(self, extractor):
        """Test enhanced pain detection with descriptors"""
        text = "Patient has sharp pain in chest and dull pain in back."
        
        entities = extractor._extract_symptoms_by_pattern(text)
        
        pain_entities = [e for e in entities if 'pain' in e.text.lower()]
        assert len(pain_entities) >= 2
        
        pain_texts = [e.text.lower() for e in pain_entities]
        assert any('sharp pain' in text for text in pain_texts)
        assert any('dull pain' in text for text in pain_texts)
    
    def test_extract_procedures_by_pattern(self, extractor):
        """Test procedure extraction"""
        text = "Patient underwent surgery and will need physical therapy. X-ray shows normal results."
        
        entities = extractor._extract_procedures_by_pattern(text)
        
        procedure_names = [e.text.lower() for e in entities]
        assert 'surgery' in procedure_names
        assert 'physical therapy' in procedure_names
        assert 'x-ray' in procedure_names
        
        for entity in entities:
            assert entity.entity_type == EntityType.PROCEDURE
    
    @pytest.mark.asyncio
    async def test_extract_with_rules_dosage(self, extractor):
        """Test rule-based extraction for dosage patterns"""
        text = "Patient takes metformin 500mg twice daily and insulin 10 units before meals."
        
        entities = await extractor._extract_with_rules(text)
        
        drug_entities = [e for e in entities if e.entity_type == EntityType.DRUG]
        drug_names = [e.text.lower() for e in drug_entities]
        
        assert 'metformin' in drug_names
        assert 'insulin' in drug_names
        
        # Check that dosage rule was used
        metformin_entity = next(e for e in drug_entities if e.text.lower() == 'metformin')
        assert metformin_entity.additional_info.get('method') == 'dosage_rule'
    
    @pytest.mark.asyncio
    async def test_extract_with_rules_diagnosis(self, extractor):
        """Test rule-based extraction for diagnosis patterns"""
        text = "Patient was diagnosed with type 2 diabetes. Also suffering from chronic pain."
        
        entities = await extractor._extract_with_rules(text)
        
        disease_entities = [e for e in entities if e.entity_type == EntityType.DISEASE]
        disease_names = [e.text.lower() for e in disease_entities]
        
        assert any('diabetes' in name for name in disease_names)
        assert any('pain' in name for name in disease_names)
    
    @pytest.mark.asyncio
    async def test_extract_with_rules_symptoms(self, extractor):
        """Test rule-based extraction for symptom patterns"""
        text = "Patient is experiencing severe headaches and feeling dizzy."
        
        entities = await extractor._extract_with_rules(text)
        
        symptom_entities = [e for e in entities if e.entity_type == EntityType.SYMPTOM]
        symptom_names = [e.text.lower() for e in symptom_entities]
        
        assert any('headache' in name for name in symptom_names)
        assert any('dizzy' in name for name in symptom_names)
    
    @pytest.mark.asyncio
    async def test_extract_with_context_medical_history(self, extractor):
        """Test context-based extraction from medical history"""
        text = "Medical history: diabetes, hypertension, and asthma. Patient is stable."
        
        entities = await extractor._extract_with_context(text)
        
        disease_entities = [e for e in entities if e.entity_type == EntityType.DISEASE]
        disease_names = [e.text.lower() for e in disease_entities]
        
        assert 'diabetes' in disease_names
        assert 'hypertension' in disease_names
        assert 'asthma' in disease_names
    
    @pytest.mark.asyncio
    async def test_extract_with_context_medications(self, extractor):
        """Test context-based extraction from medication lists"""
        text = "Current medications: aspirin, metformin, and lisinopril."
        
        entities = await extractor._extract_with_context(text)
        
        drug_entities = [e for e in entities if e.entity_type == EntityType.DRUG]
        drug_names = [e.text.lower() for e in drug_entities]
        
        assert 'aspirin' in drug_names
        assert 'metformin' in drug_names
        assert 'lisinopril' in drug_names
    
    def test_deduplicate_entities(self, extractor):
        """Test entity deduplication"""
        entities = [
            MedicalEntity("aspirin", EntityType.DRUG, confidence=0.8, source_api="test"),
            MedicalEntity("Aspirin", EntityType.DRUG, confidence=0.9, source_api="test"),  # Higher confidence
            MedicalEntity("ibuprofen", EntityType.DRUG, confidence=0.7, source_api="test")
        ]
        
        deduplicated = extractor._deduplicate_entities(entities)
        
        assert len(deduplicated) == 2  # aspirin and ibuprofen
        
        # Check that higher confidence aspirin was kept
        aspirin_entity = next(e for e in deduplicated if e.text.lower() == 'aspirin')
        assert aspirin_entity.confidence == 0.9
    
    def test_calculate_confidence_distribution(self, extractor):
        """Test confidence distribution calculation"""
        entities = [
            MedicalEntity("drug1", EntityType.DRUG, confidence=0.95, source_api="test"),  # Very high
            MedicalEntity("drug2", EntityType.DRUG, confidence=0.85, source_api="test"),  # High
            MedicalEntity("drug3", EntityType.DRUG, confidence=0.65, source_api="test"),  # Medium
            MedicalEntity("drug4", EntityType.DRUG, confidence=0.45, source_api="test")   # Low
        ]
        
        distribution = extractor._calculate_confidence_distribution(entities)
        
        assert distribution[ConfidenceLevel.VERY_HIGH.value] == 1
        assert distribution[ConfidenceLevel.HIGH.value] == 1
        assert distribution[ConfidenceLevel.MEDIUM.value] == 1
        assert distribution[ConfidenceLevel.LOW.value] == 1
    
    def test_get_entity_statistics(self, extractor):
        """Test entity statistics calculation"""
        entities = [
            MedicalEntity("aspirin", EntityType.DRUG, confidence=0.9, source_api="pattern"),
            MedicalEntity("diabetes", EntityType.DISEASE, confidence=0.8, source_api="pattern"),
            MedicalEntity("headache", EntityType.SYMPTOM, confidence=0.7, source_api="rule"),
            MedicalEntity("ibuprofen", EntityType.DRUG, confidence=0.85, source_api="context")
        ]
        
        stats = extractor.get_entity_statistics(entities)
        
        assert stats['total'] == 4
        assert stats['by_type']['drug'] == 2
        assert stats['by_type']['disease'] == 1
        assert stats['by_type']['symptom'] == 1
        assert stats['by_source']['pattern'] == 2
        assert stats['by_source']['rule'] == 1
        assert stats['by_source']['context'] == 1
        assert stats['average_confidence'] == 0.8125  # (0.9 + 0.8 + 0.7 + 0.85) / 4
        assert stats['highest_confidence'] == 0.9
        assert stats['lowest_confidence'] == 0.7
    
    def test_get_entity_statistics_empty(self, extractor):
        """Test entity statistics with empty list"""
        stats = extractor.get_entity_statistics([])
        assert stats['total'] == 0
    
    @pytest.mark.asyncio
    async def test_validate_entities(self, extractor):
        """Test entity validation"""
        entities = [
            MedicalEntity("aspirin", EntityType.DRUG, confidence=0.9, source_api="test"),
            MedicalEntity("a", EntityType.DRUG, confidence=0.8, source_api="test"),  # Too short
            MedicalEntity("the", EntityType.DRUG, confidence=0.7, source_api="test"),  # Common word
            MedicalEntity("diabetes", EntityType.DISEASE, confidence=0.85, source_api="test")
        ]
        
        validated = await extractor.validate_entities(entities)
        
        assert len(validated) == 2  # Only aspirin and diabetes should remain
        validated_texts = [e.text.lower() for e in validated]
        assert 'aspirin' in validated_texts
        assert 'diabetes' in validated_texts
        assert 'a' not in validated_texts
        assert 'the' not in validated_texts
    
    @pytest.mark.asyncio
    async def test_validate_entities_with_external_validator(self, extractor):
        """Test entity validation with external validator"""
        async def mock_validator(entity):
            # Only validate entities starting with 'a'
            return entity.text.lower().startswith('a')
        
        entities = [
            MedicalEntity("aspirin", EntityType.DRUG, confidence=0.9, source_api="test"),
            MedicalEntity("diabetes", EntityType.DISEASE, confidence=0.8, source_api="test")
        ]
        
        validated = await extractor.validate_entities(entities, mock_validator)
        
        assert len(validated) == 1
        assert validated[0].text == "aspirin"
    
    @pytest.mark.asyncio
    async def test_extract_entities_full_workflow(self, extractor):
        """Test complete entity extraction workflow"""
        text = """
        Patient John Doe, 45 years old, presents with complaints of severe headache and nausea.
        Medical history includes diabetes and hypertension.
        Current medications: metformin 500mg twice daily, lisinopril 10mg once daily.
        Patient was diagnosed with migraine and prescribed sumatriptan.
        Recommended follow-up with neurologist and MRI scan.
        """
        
        result = await extractor.extract_entities(text)
        
        assert isinstance(result, ExtractionResult)
        assert len(result.entities) > 0
        assert result.processing_time > 0
        assert result.method_used == "hybrid_pattern_rule_context"
        assert result.text_length == len(text)
        
        # Check that different entity types were found
        entity_types = {e.entity_type for e in result.entities}
        assert EntityType.DRUG in entity_types
        assert EntityType.DISEASE in entity_types
        assert EntityType.SYMPTOM in entity_types
        
        # Check specific entities
        entity_texts = [e.text.lower() for e in result.entities]
        assert any('headache' in text for text in entity_texts)
        assert any('diabetes' in text for text in entity_texts)
        assert any('metformin' in text for text in entity_texts)
    
    @pytest.mark.asyncio
    async def test_extract_entities_with_confidence_threshold(self, extractor):
        """Test entity extraction with confidence threshold"""
        text = "Patient has diabetes and takes aspirin."
        
        # High threshold should return fewer entities
        result_high = await extractor.extract_entities(text, confidence_threshold=0.9)
        result_low = await extractor.extract_entities(text, confidence_threshold=0.5)
        
        assert len(result_high.entities) <= len(result_low.entities)
    
    @pytest.mark.asyncio
    async def test_extract_entities_specific_types(self, extractor):
        """Test entity extraction for specific entity types"""
        text = "Patient has diabetes, takes aspirin, and complains of headache."
        
        # Extract only drugs
        result_drugs = await extractor.extract_entities(text, entity_types=['drug'])
        drug_entities = [e for e in result_drugs.entities if e.entity_type == EntityType.DRUG]
        non_drug_entities = [e for e in result_drugs.entities if e.entity_type != EntityType.DRUG]
        
        assert len(drug_entities) > 0
        assert len(non_drug_entities) == 0  # Should only find drugs
        
        # Extract only diseases
        result_diseases = await extractor.extract_entities(text, entity_types=['disease'])
        disease_entities = [e for e in result_diseases.entities if e.entity_type == EntityType.DISEASE]
        
        assert len(disease_entities) > 0
    
    @pytest.mark.asyncio
    async def test_extract_entities_empty_text(self, extractor):
        """Test entity extraction with empty text"""
        result = await extractor.extract_entities("")
        
        assert len(result.entities) == 0
        assert result.processing_time >= 0
    
    @pytest.mark.asyncio
    async def test_extract_entities_no_medical_content(self, extractor):
        """Test entity extraction with non-medical text"""
        text = "The weather is nice today. I went to the store and bought groceries."
        
        result = await extractor.extract_entities(text)
        
        # Should find very few or no medical entities
        assert len(result.entities) == 0 or all(e.confidence < 0.7 for e in result.entities)
from src
.medical_nlp import DocumentSummarizer

class TestDocumentSummarizer:
    """Test cases for DocumentSummarizer class"""
    
    @pytest.fixture
    def summarizer(self):
        """Create DocumentSummarizer instance for testing"""
        return DocumentSummarizer()
    
    def test_summarizer_initialization(self, summarizer):
        """Test summarizer initialization"""
        assert summarizer.entity_extractor is not None
        assert summarizer.document_patterns is not None
        assert summarizer.section_patterns is not None
        
        # Check that patterns contain expected data
        assert 'prescription' in summarizer.document_patterns
        assert 'lab_report' in summarizer.document_patterns
        assert 'chief_complaint' in summarizer.section_patterns
    
    def test_detect_document_type_prescription(self, summarizer):
        """Test prescription document type detection"""
        text = """
        Rx: Metformin 500mg
        Take twice daily with meals
        Dispense: 60 tablets
        Refills: 2
        """
        
        doc_type = summarizer._detect_document_type(text)
        assert doc_type == 'prescription'
    
    def test_detect_document_type_lab_report(self, summarizer):
        """Test lab report document type detection"""
        text = """
        Laboratory Results
        Glucose: 120 mg/dL (Normal range: 70-100)
        Hemoglobin: 14.2 g/dL (Normal)
        Specimen collected: 2024-01-15
        """
        
        doc_type = summarizer._detect_document_type(text)
        assert doc_type == 'lab_report'
    
    def test_detect_document_type_medical_note(self, summarizer):
        """Test medical note document type detection"""
        text = """
        Chief Complaint: Chest pain
        History of Present Illness: Patient presents with acute chest pain
        Physical Exam: Normal heart sounds
        Assessment: Possible angina
        Plan: EKG and cardiac enzymes
        """
        
        doc_type = summarizer._detect_document_type(text)
        assert doc_type == 'medical_note'
    
    def test_detect_document_type_unknown(self, summarizer):
        """Test unknown document type detection"""
        text = "This is just regular text with no medical content."
        
        doc_type = summarizer._detect_document_type(text)
        assert doc_type == 'unknown'
    
    def test_extract_sections(self, summarizer):
        """Test section extraction from medical text"""
        text = """
        Chief Complaint: Severe headache
        History of Present Illness: Patient has had headaches for 3 days.
        Past Medical History: Diabetes, hypertension
        Medications: Metformin, Lisinopril
        Assessment: Migraine headache
        Plan: Prescribe sumatriptan
        """
        
        sections = summarizer._extract_sections(text)
        
        assert 'chief_complaint' in sections
        assert 'history_present_illness' in sections
        assert 'past_medical_history' in sections
        assert 'medications' in sections
        assert 'assessment' in sections
        assert 'plan' in sections
        
        assert 'severe headache' in sections['chief_complaint'].lower()
        assert 'diabetes' in sections['past_medical_history'].lower()
    
    def test_extract_medications_from_text(self, summarizer):
        """Test medication extraction from text"""
        text = """
        Patient is taking metformin 500mg twice daily,
        lisinopril 10mg once daily, and aspirin 81mg daily.
        Also prescribed amoxicillin 250mg three times daily.
        """
        
        medications = summarizer._extract_medications_from_text(text)
        
        med_names = [med.lower() for med in medications]
        assert 'metformin' in med_names
        assert 'lisinopril' in med_names
        assert 'aspirin' in med_names
        assert 'amoxicillin' in med_names
    
    def test_extract_abnormal_lab_values(self, summarizer):
        """Test abnormal lab value extraction"""
        text = """
        Lab Results:
        Glucose: High (150 mg/dL)
        Hemoglobin: Low (10.2 g/dL)
        Cholesterol: Elevated (250 mg/dL)
        Creatinine: Normal (1.0 mg/dL)
        """
        
        abnormal_values = summarizer._extract_abnormal_lab_values(text)
        
        assert len(abnormal_values) >= 3
        abnormal_text = ' '.join(abnormal_values).lower()
        assert 'glucose' in abnormal_text
        assert 'hemoglobin' in abnormal_text
        assert 'cholesterol' in abnormal_text
    
    def test_extract_diagnoses_from_text(self, summarizer):
        """Test diagnosis extraction from assessment text"""
        text = """
        Assessment:
        1. Type 2 diabetes mellitus
        2. Hypertension, uncontrolled
        3. Hyperlipidemia
        Also considering pneumonia based on symptoms.
        """
        
        diagnoses = summarizer._extract_diagnoses_from_text(text)
        
        diagnosis_text = ' '.join(diagnoses).lower()
        assert 'diabetes' in diagnosis_text
        assert 'hypertension' in diagnosis_text
        assert 'pneumonia' in diagnosis_text
    
    def test_extract_critical_terms(self, summarizer):
        """Test critical term extraction"""
        text = """
        Patient presents with acute chest pain.
        EKG shows abnormal findings.
        Troponin levels are critical.
        Urgent cardiology consultation needed.
        """
        
        critical_terms = summarizer._extract_critical_terms(text)
        
        critical_text = ' '.join(critical_terms).lower()
        assert 'acute' in critical_text
        assert 'abnormal' in critical_text
        assert 'critical' in critical_text
        assert 'urgent' in critical_text
    
    def test_summarize_prescription(self, summarizer):
        """Test prescription document summarization"""
        text = """
        Prescription for John Doe
        Rx: Metformin 500mg - Take twice daily with meals
        Rx: Lisinopril 10mg - Take once daily in morning
        Dispense: 30-day supply
        Refills: 2
        Allergies: Penicillin
        """
        
        sections = summarizer._extract_sections(text)
        summary = summarizer._summarize_prescription(text, sections, 'detailed')
        
        assert 'metformin' in summary.lower()
        assert 'lisinopril' in summary.lower()
        assert 'medication' in summary.lower()
    
    def test_summarize_lab_report(self, summarizer):
        """Test lab report summarization"""
        text = """
        Laboratory Report
        Patient: Jane Smith
        Date: 2024-01-15
        
        Results:
        Glucose: 180 mg/dL (High) - Reference: 70-100
        Hemoglobin: 8.5 g/dL (Low) - Reference: 12-16
        Cholesterol: 220 mg/dL (Normal) - Reference: <200
        """
        
        sections = summarizer._extract_sections(text)
        summary = summarizer._summarize_lab_report(text, sections, 'detailed')
        
        assert 'abnormal' in summary.lower() or 'glucose' in summary.lower()
        assert 'lab' in summary.lower() or 'result' in summary.lower()
    
    def test_summarize_medical_note(self, summarizer):
        """Test medical note summarization"""
        text = """
        Chief Complaint: Chest pain and shortness of breath
        
        History of Present Illness: 
        45-year-old male presents with acute onset chest pain.
        
        Physical Exam:
        Vital signs stable. Heart rate 85, BP 140/90.
        
        Assessment: 
        Possible acute coronary syndrome
        
        Plan:
        1. EKG
        2. Cardiac enzymes
        3. Chest X-ray
        """
        
        sections = summarizer._extract_sections(text)
        summary = summarizer._summarize_medical_note(text, sections, 'detailed')
        
        assert 'chest pain' in summary.lower()
        assert 'assessment' in summary.lower() or 'coronary' in summary.lower()
        assert 'plan' in summary.lower() or 'ekg' in summary.lower()
    
    def test_split_into_sentences(self, summarizer):
        """Test sentence splitting"""
        text = "This is sentence one. This is sentence two! Is this sentence three? Yes it is."
        
        sentences = summarizer._split_into_sentences(text)
        
        assert len(sentences) == 4
        assert "This is sentence one" in sentences[0]
        assert "This is sentence two" in sentences[1]
    
    def test_score_sentences_for_relevance(self, summarizer):
        """Test sentence relevance scoring"""
        sentences = [
            "The weather is nice today.",
            "Patient has diabetes and takes medication.",
            "Treatment plan includes therapy and follow-up.",
            "I went to the store yesterday."
        ]
        
        scored = summarizer._score_sentences_for_relevance(sentences)
        
        # Medical sentences should have higher scores
        medical_scores = [score for sentence, score, idx in scored if 'patient' in sentence.lower() or 'treatment' in sentence.lower()]
        non_medical_scores = [score for sentence, score, idx in scored if 'weather' in sentence.lower() or 'store' in sentence.lower()]
        
        assert max(medical_scores) > max(non_medical_scores)
    
    def test_calculate_summary_confidence(self, summarizer):
        """Test summary confidence calculation"""
        original_text = "This is a medical document with patient information."
        summary = "Patient information summary with key details."
        sections = {'chief_complaint': 'test', 'assessment': 'test'}
        
        # Known document type should have higher confidence
        confidence_known = summarizer._calculate_summary_confidence(
            original_text, summary, sections, 'medical_note'
        )
        
        confidence_unknown = summarizer._calculate_summary_confidence(
            original_text, summary, {}, 'unknown'
        )
        
        assert confidence_known > confidence_unknown
        assert 0.0 <= confidence_known <= 1.0
        assert 0.0 <= confidence_unknown <= 1.0
    
    @pytest.mark.asyncio
    async def test_summarize_document_full_workflow(self, summarizer):
        """Test complete document summarization workflow"""
        text = """
        Chief Complaint: Severe headache and nausea
        
        History of Present Illness:
        Patient is a 35-year-old female presenting with severe headache
        that started 2 days ago. Associated with nausea and photophobia.
        
        Past Medical History: Migraine headaches, depression
        
        Medications: Sumatriptan as needed, sertraline 50mg daily
        
        Physical Exam:
        Vital signs: BP 120/80, HR 72, Temp 98.6F
        Neurological exam normal
        
        Assessment:
        Migraine headache, acute episode
        
        Plan:
        1. Continue sumatriptan as needed
        2. Consider preventive therapy
        3. Follow up in 2 weeks
        """
        
        result = await summarizer.summarize_document(text, summary_length='detailed')
        
        assert result['document_type'] == 'medical_note'
        assert len(result['summary']) > 0
        assert len(result['key_findings']) > 0
        assert result['confidence_score'] > 0
        assert result['processing_time'] > 0
        assert 'sections_found' in result
        assert 'metadata' in result
        
        # Check that key information is in summary
        summary_lower = result['summary'].lower()
        assert 'headache' in summary_lower
        assert 'migraine' in summary_lower
    
    @pytest.mark.asyncio
    async def test_summarize_document_different_lengths(self, summarizer):
        """Test document summarization with different length settings"""
        text = """
        Patient presents with chest pain. Has history of diabetes.
        Physical exam shows elevated blood pressure. EKG is normal.
        Plan includes cardiac enzymes and chest X-ray. Follow up needed.
        """
        
        brief_result = await summarizer.summarize_document(text, summary_length='brief')
        detailed_result = await summarizer.summarize_document(text, summary_length='detailed')
        comprehensive_result = await summarizer.summarize_document(text, summary_length='comprehensive')
        
        # Brief should be shorter than detailed, detailed shorter than comprehensive
        assert len(brief_result['summary']) <= len(detailed_result['summary'])
        assert len(detailed_result['summary']) <= len(comprehensive_result['summary'])
    
    @pytest.mark.asyncio
    async def test_summarize_document_without_entities(self, summarizer):
        """Test document summarization without entity extraction"""
        text = "Patient has diabetes and takes metformin."
        
        result = await summarizer.summarize_document(text, extract_entities=False)
        
        assert len(result['extracted_entities']) == 0
    
    @pytest.mark.asyncio
    async def test_summarize_document_empty_text(self, summarizer):
        """Test document summarization with empty text"""
        result = await summarizer.summarize_document("")
        
        assert result['original_length'] == 0
        assert result['document_type'] == 'unknown'
        assert len(result['summary']) > 0  # Should have some default summary
    
    @pytest.mark.asyncio
    async def test_summarize_document_prescription_type(self, summarizer):
        """Test summarization of prescription document"""
        text = """
        Prescription for Patient: John Smith
        DOB: 01/01/1980
        
        Rx: Metformin 500mg
        Sig: Take 1 tablet twice daily with meals
        Dispense: 60 tablets
        Refills: 2
        
        Rx: Lisinopril 10mg
        Sig: Take 1 tablet daily in morning
        Dispense: 30 tablets
        Refills: 3
        
        Allergies: Penicillin, Sulfa drugs
        """
        
        result = await summarizer.summarize_document(text, document_type='prescription')
        
        assert result['document_type'] == 'prescription'
        summary_lower = result['summary'].lower()
        assert 'metformin' in summary_lower
        assert 'lisinopril' in summary_lower
        assert 'medication' in summary_lower or 'prescribed' in summary_lower