"""
Unit tests for document processor
"""

import pytest
from unittest.mock import Mock, patch, mock_open
import io
from src.document_processor import DocumentProcessor, ProcessedDocument, SupportedFormat

class TestDocumentProcessor:
    """Test cases for DocumentProcessor class"""
    
    @pytest.fixture
    def processor(self):
        """Create DocumentProcessor instance for testing"""
        return DocumentProcessor()
    
    def test_processor_initialization(self, processor):
        """Test processor initialization"""
        assert processor.document_summarizer is not None
        assert processor.max_file_size == 10 * 1024 * 1024  # 10MB
        assert len(processor.supported_extensions) == 5
        assert '.pdf' in processor.supported_extensions
        assert '.txt' in processor.supported_extensions
        assert '.docx' in processor.supported_extensions
    
    def test_validate_file_success(self, processor):
        """Test successful file validation"""
        result = processor._validate_file("test.txt", 1000)
        
        assert result['valid'] is True
    
    def test_validate_file_too_large(self, processor):
        """Test file validation with oversized file"""
        large_size = processor.max_file_size + 1
        result = processor._validate_file("test.txt", large_size)
        
        assert result['valid'] is False
        assert 'exceeds maximum' in result['error']
    
    def test_validate_file_unsupported_format(self, processor):
        """Test file validation with unsupported format"""
        result = processor._validate_file("test.xyz", 1000)
        
        assert result['valid'] is False
        assert 'Unsupported file format' in result['error']
    
    def test_validate_file_unsafe_filename(self, processor):
        """Test file validation with unsafe filename"""
        result = processor._validate_file("../../../etc/passwd", 1000)
        
        assert result['valid'] is False
        assert 'unsafe characters' in result['error']
    
    def test_is_safe_filename(self, processor):
        """Test filename safety check"""
        assert processor._is_safe_filename("document.pdf") is True
        assert processor._is_safe_filename("medical_report_2024.txt") is True
        assert processor._is_safe_filename("../malicious.txt") is False
        assert processor._is_safe_filename("file/with/path.pdf") is False
        assert processor._is_safe_filename("file:with:colons.txt") is False
        assert processor._is_safe_filename("a" * 300 + ".txt") is False  # Too long
    
    def test_detect_file_format_by_extension(self, processor):
        """Test file format detection by extension"""
        assert processor._detect_file_format("test.pdf", b"dummy") == SupportedFormat.PDF
        assert processor._detect_file_format("test.txt", b"dummy") == SupportedFormat.TXT
        assert processor._detect_file_format("test.docx", b"dummy") == SupportedFormat.DOCX
        assert processor._detect_file_format("test.doc", b"dummy") == SupportedFormat.DOC
        assert processor._detect_file_format("test.rtf", b"dummy") == SupportedFormat.RTF
    
    def test_detect_by_signature_pdf(self, processor):
        """Test PDF detection by signature"""
        pdf_header = b'%PDF-1.4\n'
        format_detected = processor._detect_by_signature(pdf_header)
        
        assert format_detected == SupportedFormat.PDF
    
    def test_detect_by_signature_docx(self, processor):
        """Test DOCX detection by signature (ZIP)"""
        zip_header = b'PK\x03\x04'
        format_detected = processor._detect_by_signature(zip_header)
        
        assert format_detected == SupportedFormat.DOCX
    
    def test_detect_by_signature_rtf(self, processor):
        """Test RTF detection by signature"""
        rtf_header = b'{\\rtf1\\ansi'
        format_detected = processor._detect_by_signature(rtf_header)
        
        assert format_detected == SupportedFormat.RTF
    
    def test_detect_by_signature_txt(self, processor):
        """Test plain text detection"""
        text_content = b'This is plain text content'
        format_detected = processor._detect_by_signature(text_content)
        
        assert format_detected == SupportedFormat.TXT
    
    @pytest.mark.asyncio
    async def test_extract_text_from_txt(self, processor):
        """Test text extraction from plain text file"""
        text_content = b"This is a medical report.\nPatient has diabetes."
        
        extracted = await processor._extract_text_from_txt(text_content)
        
        assert "medical report" in extracted
        assert "diabetes" in extracted
    
    @pytest.mark.asyncio
    async def test_extract_text_from_txt_different_encodings(self, processor):
        """Test text extraction with different encodings"""
        # UTF-8 content
        utf8_content = "Patient has café syndrome".encode('utf-8')
        extracted = await processor._extract_text_from_txt(utf8_content)
        assert "café" in extracted
        
        # Latin-1 content
        latin1_content = "Patient has café syndrome".encode('latin-1')
        extracted = await processor._extract_text_from_txt(latin1_content)
        assert "caf" in extracted  # Should handle encoding gracefully
    
    @pytest.mark.asyncio
    async def test_extract_text_from_txt_file_object(self, processor):
        """Test text extraction from file-like object"""
        text_content = "Medical report content"
        file_obj = io.StringIO(text_content)
        
        extracted = await processor._extract_text_from_txt(file_obj)
        
        assert extracted == text_content
    
    @pytest.mark.asyncio
    async def test_extract_text_from_pdf_mock(self, processor):
        """Test PDF text extraction with mocked PyPDF2"""
        pdf_content = b'%PDF-1.4 mock content'
        
        # Mock PyPDF2
        mock_page = Mock()
        mock_page.extract_text.return_value = "Extracted PDF text content"
        
        mock_reader = Mock()
        mock_reader.pages = [mock_page]
        
        with patch('src.document_processor.PyPDF2.PdfReader', return_value=mock_reader):
            extracted = await processor._extract_text_from_pdf(pdf_content)
        
        assert extracted == "Extracted PDF text content"
    
    @pytest.mark.asyncio
    async def test_extract_text_from_pdf_no_library(self, processor):
        """Test PDF extraction when PyPDF2 is not available"""
        pdf_content = b'%PDF-1.4 mock content'
        
        with patch('src.document_processor.PyPDF2', side_effect=ImportError()):
            extracted = await processor._extract_text_from_pdf(pdf_content)
        
        assert "PDF processing library not available" in extracted
    
    @pytest.mark.asyncio
    async def test_extract_text_from_docx_mock(self, processor):
        """Test DOCX text extraction with mocked python-docx"""
        docx_content = b'PK\x03\x04 mock docx'
        
        # Mock python-docx
        mock_paragraph = Mock()
        mock_paragraph.text = "Paragraph text from DOCX"
        
        mock_doc = Mock()
        mock_doc.paragraphs = [mock_paragraph]
        
        with patch('src.document_processor.python_docx.Document', return_value=mock_doc):
            extracted = await processor._extract_text_from_docx(docx_content)
        
        assert extracted == "Paragraph text from DOCX"
    
    @pytest.mark.asyncio
    async def test_extract_text_from_docx_no_library(self, processor):
        """Test DOCX extraction when python-docx is not available"""
        docx_content = b'PK\x03\x04 mock docx'
        
        with patch('src.document_processor.python_docx', side_effect=ImportError()):
            extracted = await processor._extract_text_from_docx(docx_content)
        
        assert "DOCX processing library not available" in extracted
    
    @pytest.mark.asyncio
    async def test_extract_text_from_doc(self, processor):
        """Test DOC text extraction (placeholder implementation)"""
        doc_content = b'\xd0\xcf\x11\xe0 mock doc'
        
        extracted = await processor._extract_text_from_doc(doc_content)
        
        assert "DOC file detected" in extracted
        assert "not fully implemented" in extracted
    
    @pytest.mark.asyncio
    async def test_extract_text_from_rtf(self, processor):
        """Test RTF text extraction"""
        rtf_content = b'{\\rtf1\\ansi Patient has \\b diabetes\\b0 .}'
        
        extracted = await processor._extract_text_from_rtf(rtf_content)
        
        assert "Patient has" in extracted
        assert "diabetes" in extracted
        # RTF control codes should be removed
        assert "\\rtf1" not in extracted
        assert "\\b" not in extracted
    
    def test_calculate_extraction_confidence(self, processor):
        """Test extraction confidence calculation"""
        # Long medical text should have high confidence
        long_medical_text = "Patient presents with diabetes and hypertension. Treatment plan includes medication and lifestyle changes." * 3
        confidence_high = processor._calculate_extraction_confidence(
            long_medical_text, SupportedFormat.TXT, "medical.txt"
        )
        
        # Short non-medical text should have lower confidence
        short_text = "Hello world"
        confidence_low = processor._calculate_extraction_confidence(
            short_text, SupportedFormat.PDF, "document.pdf"
        )
        
        assert confidence_high > confidence_low
        assert 0.0 <= confidence_high <= 1.0
        assert 0.0 <= confidence_low <= 1.0
    
    def test_calculate_extraction_confidence_error_text(self, processor):
        """Test confidence calculation with error text"""
        error_text = "Error extracting content from file"
        confidence = processor._calculate_extraction_confidence(
            error_text, SupportedFormat.PDF, "test.pdf"
        )
        
        # Should have reduced confidence due to error
        assert confidence < 0.5
    
    @pytest.mark.asyncio
    async def test_process_uploaded_file_success(self, processor):
        """Test successful file processing"""
        file_content = b"Patient medical report with diabetes diagnosis."
        filename = "medical_report.txt"
        file_size = len(file_content)
        
        result = await processor.process_uploaded_file(file_content, filename, file_size)
        
        assert isinstance(result, ProcessedDocument)
        assert result.filename == filename
        assert result.file_size == file_size
        assert result.file_type == "txt"
        assert "diabetes" in result.extracted_text
        assert result.text_length > 0
        assert result.processing_time > 0
        assert result.confidence_score > 0
        assert result.error is None
        assert result.metadata is not None
    
    @pytest.mark.asyncio
    async def test_process_uploaded_file_validation_error(self, processor):
        """Test file processing with validation error"""
        file_content = b"content"
        filename = "test.xyz"  # Unsupported format
        
        result = await processor.process_uploaded_file(file_content, filename)
        
        assert result.error is not None
        assert "Unsupported file format" in result.error
        assert result.extracted_text == ""
        assert result.confidence_score == 0.0
    
    @pytest.mark.asyncio
    async def test_process_uploaded_file_extraction_error(self, processor):
        """Test file processing with extraction error"""
        file_content = b"content"
        filename = "test.pdf"
        
        # Mock extraction to raise an exception
        with patch.object(processor, '_extract_text_from_file', side_effect=Exception("Extraction failed")):
            result = await processor.process_uploaded_file(file_content, filename)
        
        assert result.error is not None
        assert "Extraction failed" in result.error
    
    @pytest.mark.asyncio
    async def test_process_and_summarize_document_success(self, processor):
        """Test combined document processing and summarization"""
        file_content = b"""
        Chief Complaint: Patient presents with chest pain
        History: 45-year-old male with diabetes
        Assessment: Possible cardiac event
        Plan: EKG and cardiac enzymes
        """
        filename = "medical_note.txt"
        
        # Mock the summarizer
        mock_summary_result = {
            'document_type': 'medical_note',
            'summary': 'Patient with chest pain and diabetes',
            'key_findings': ['chest pain', 'diabetes'],
            'processing_time': 0.1
        }
        
        with patch.object(processor.document_summarizer, 'summarize_document', return_value=mock_summary_result):
            result = await processor.process_and_summarize_document(file_content, filename)
        
        assert result['success'] is True
        assert 'document_info' in result
        assert 'summary_info' in result
        assert result['document_info']['filename'] == filename
        assert result['summary_info']['document_type'] == 'medical_note'
    
    @pytest.mark.asyncio
    async def test_process_and_summarize_document_processing_error(self, processor):
        """Test combined processing with document processing error"""
        file_content = b"content"
        filename = "test.xyz"  # Invalid format
        
        result = await processor.process_and_summarize_document(file_content, filename)
        
        assert result['success'] is False
        assert 'error' in result
        assert 'document_info' in result
    
    @pytest.mark.asyncio
    async def test_process_and_summarize_document_summarization_error(self, processor):
        """Test combined processing with summarization error"""
        file_content = b"Valid medical text content"
        filename = "test.txt"
        
        # Mock summarizer to raise exception
        with patch.object(processor.document_summarizer, 'summarize_document', side_effect=Exception("Summary failed")):
            result = await processor.process_and_summarize_document(file_content, filename)
        
        assert result['success'] is False
        assert 'summarization failed' in result['error']
        assert 'document_info' in result
        assert 'extracted_text' in result
    
    def test_get_supported_formats(self, processor):
        """Test getting supported formats information"""
        formats_info = processor.get_supported_formats()
        
        assert 'supported_extensions' in formats_info
        assert 'supported_formats' in formats_info
        assert 'max_file_size' in formats_info
        assert 'max_file_size_mb' in formats_info
        assert 'format_descriptions' in formats_info
        
        assert '.pdf' in formats_info['supported_extensions']
        assert 'pdf' in formats_info['supported_formats']
        assert formats_info['max_file_size_mb'] == 10
    
    def test_validate_file_before_upload(self, processor):
        """Test pre-upload file validation"""
        # Valid file
        result_valid = processor.validate_file_before_upload("test.txt", 1000)
        assert result_valid['valid'] is True
        
        # Invalid file
        result_invalid = processor.validate_file_before_upload("test.xyz", 1000)
        assert result_invalid['valid'] is False
    
    @pytest.mark.asyncio
    async def test_extract_text_only(self, processor):
        """Test text-only extraction without summarization"""
        file_content = b"Medical report with patient information"
        filename = "report.txt"
        
        result = await processor.extract_text_only(file_content, filename)
        
        assert result['success'] is True
        assert 'extracted_text' in result
        assert 'metadata' in result
        assert result['extracted_text'] == "Medical report with patient information"
        assert result['metadata']['filename'] == filename
        assert result['metadata']['file_type'] == 'txt'
    
    @pytest.mark.asyncio
    async def test_extract_text_only_with_error(self, processor):
        """Test text-only extraction with error"""
        file_content = b"content"
        filename = "test.xyz"  # Invalid format
        
        result = await processor.extract_text_only(file_content, filename)
        
        assert result['success'] is False
        assert result['error'] is not None
        assert result['extracted_text'] == ""


class TestProcessedDocument:
    """Test cases for ProcessedDocument dataclass"""
    
    def test_processed_document_creation(self):
        """Test creating ProcessedDocument"""
        doc = ProcessedDocument(
            filename="test.txt",
            file_size=1000,
            file_type="txt",
            extracted_text="Test content",
            text_length=12,
            processing_time=0.5,
            confidence_score=0.8,
            metadata={'word_count': 2}
        )
        
        assert doc.filename == "test.txt"
        assert doc.file_size == 1000
        assert doc.file_type == "txt"
        assert doc.extracted_text == "Test content"
        assert doc.text_length == 12
        assert doc.processing_time == 0.5
        assert doc.confidence_score == 0.8
        assert doc.error is None
        assert doc.metadata['word_count'] == 2
    
    def test_processed_document_with_error(self):
        """Test ProcessedDocument with error"""
        doc = ProcessedDocument(
            filename="test.txt",
            file_size=0,
            file_type="unknown",
            extracted_text="",
            text_length=0,
            processing_time=0.1,
            confidence_score=0.0,
            error="Processing failed"
        )
        
        assert doc.error == "Processing failed"
        assert doc.confidence_score == 0.0


class TestSupportedFormat:
    """Test cases for SupportedFormat enum"""
    
    def test_supported_format_values(self):
        """Test SupportedFormat enum values"""
        assert SupportedFormat.PDF.value == "pdf"
        assert SupportedFormat.TXT.value == "txt"
        assert SupportedFormat.DOC.value == "doc"
        assert SupportedFormat.DOCX.value == "docx"
        assert SupportedFormat.RTF.value == "rtf"
    
    def test_supported_format_count(self):
        """Test that we have the expected number of supported formats"""
        formats = list(SupportedFormat)
        assert len(formats) == 5