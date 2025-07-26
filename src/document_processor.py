"""
Document upload and processing capabilities for Intelligent Healthcare Navigator
Handles file upload, text extraction, and format validation
"""

import os
import tempfile
from typing import Dict, Any, List, Optional, BinaryIO, Union
from dataclasses import dataclass
from enum import Enum
import mimetypes
import hashlib
from pathlib import Path

from src.medical_nlp import DocumentSummarizer
from src.utils import setup_logging, validate_file_type

logger = setup_logging()

class SupportedFormat(Enum):
    """Supported document formats"""
    PDF = "pdf"
    TXT = "txt"
    DOC = "doc"
    DOCX = "docx"
    RTF = "rtf"

@dataclass
class ProcessedDocument:
    """Container for processed document information"""
    filename: str
    file_size: int
    file_type: str
    extracted_text: str
    text_length: int
    processing_time: float
    confidence_score: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

class DocumentProcessor:
    """Handles document upload and text extraction"""
    
    def __init__(self, document_summarizer: DocumentSummarizer = None):
        """Initialize document processor"""
        self.document_summarizer = document_summarizer or DocumentSummarizer()
        
        # Maximum file size (10MB)
        self.max_file_size = 10 * 1024 * 1024
        
        # Supported file extensions
        self.supported_extensions = {
            '.pdf': SupportedFormat.PDF,
            '.txt': SupportedFormat.TXT,
            '.doc': SupportedFormat.DOC,
            '.docx': SupportedFormat.DOCX,
            '.rtf': SupportedFormat.RTF
        }
        
        # MIME type mapping
        self.mime_type_mapping = {
            'application/pdf': SupportedFormat.PDF,
            'text/plain': SupportedFormat.TXT,
            'application/msword': SupportedFormat.DOC,
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': SupportedFormat.DOCX,
            'application/rtf': SupportedFormat.RTF,
            'text/rtf': SupportedFormat.RTF
        }
        
        logger.info("Document processor initialized")
    
    async def process_uploaded_file(
        self, 
        file_data: Union[bytes, BinaryIO], 
        filename: str,
        file_size: Optional[int] = None
    ) -> ProcessedDocument:
        """Process uploaded file and extract text"""
        import time
        start_time = time.time()
        
        logger.info(f"Processing uploaded file: {filename} ({file_size} bytes)")
        
        try:
            # Validate file
            validation_result = self._validate_file(filename, file_size or len(file_data) if isinstance(file_data, bytes) else 0)
            if not validation_result['valid']:
                return ProcessedDocument(
                    filename=filename,
                    file_size=file_size or 0,
                    file_type="unknown",
                    extracted_text="",
                    text_length=0,
                    processing_time=time.time() - start_time,
                    confidence_score=0.0,
                    error=validation_result['error']
                )
            
            # Determine file format
            file_format = self._detect_file_format(filename, file_data)
            
            # Extract text based on format
            extracted_text = await self._extract_text_from_file(file_data, file_format, filename)
            
            # Calculate confidence score
            confidence_score = self._calculate_extraction_confidence(
                extracted_text, file_format, filename
            )
            
            processing_time = time.time() - start_time
            
            result = ProcessedDocument(
                filename=filename,
                file_size=file_size or (len(file_data) if isinstance(file_data, bytes) else 0),
                file_type=file_format.value if file_format else "unknown",
                extracted_text=extracted_text,
                text_length=len(extracted_text),
                processing_time=processing_time,
                confidence_score=confidence_score,
                metadata={
                    'word_count': len(extracted_text.split()) if extracted_text else 0,
                    'line_count': len(extracted_text.splitlines()) if extracted_text else 0,
                    'character_count': len(extracted_text) if extracted_text else 0
                }
            )
            
            logger.info(f"File processed successfully: {len(extracted_text)} chars extracted in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}")
            return ProcessedDocument(
                filename=filename,
                file_size=file_size or 0,
                file_type="unknown",
                extracted_text="",
                text_length=0,
                processing_time=time.time() - start_time,
                confidence_score=0.0,
                error=str(e)
            )
    
    def _validate_file(self, filename: str, file_size: int) -> Dict[str, Any]:
        """Validate uploaded file"""
        # Check file size
        if file_size > self.max_file_size:
            return {
                'valid': False,
                'error': f'File size ({file_size} bytes) exceeds maximum allowed size ({self.max_file_size} bytes)'
            }
        
        # Check file extension
        file_extension = Path(filename).suffix.lower()
        if file_extension not in self.supported_extensions:
            supported_exts = ', '.join(self.supported_extensions.keys())
            return {
                'valid': False,
                'error': f'Unsupported file format. Supported formats: {supported_exts}'
            }
        
        # Check filename for security
        if not self._is_safe_filename(filename):
            return {
                'valid': False,
                'error': 'Invalid filename. Filename contains unsafe characters.'
            }
        
        return {'valid': True}
    
    def _is_safe_filename(self, filename: str) -> bool:
        """Check if filename is safe (no path traversal, etc.)"""
        # Remove any path components
        safe_filename = os.path.basename(filename)
        
        # Check for dangerous characters
        dangerous_chars = ['..', '/', '\\', ':', '*', '?', '"', '<', '>', '|']
        for char in dangerous_chars:
            if char in filename:
                return False
        
        # Check length
        if len(safe_filename) > 255:
            return False
        
        return True
    
    def _detect_file_format(self, filename: str, file_data: Union[bytes, BinaryIO]) -> Optional[SupportedFormat]:
        """Detect file format from extension and content"""
        # First try by extension
        file_extension = Path(filename).suffix.lower()
        if file_extension in self.supported_extensions:
            format_by_ext = self.supported_extensions[file_extension]
        else:
            format_by_ext = None
        
        # Try by MIME type if we have bytes
        format_by_mime = None
        if isinstance(file_data, bytes):
            mime_type, _ = mimetypes.guess_type(filename)
            if mime_type in self.mime_type_mapping:
                format_by_mime = self.mime_type_mapping[mime_type]
        
        # Try by content signature (magic bytes)
        format_by_signature = self._detect_by_signature(file_data)
        
        # Prefer signature detection, then extension, then MIME type
        return format_by_signature or format_by_ext or format_by_mime
    
    def _detect_by_signature(self, file_data: Union[bytes, BinaryIO]) -> Optional[SupportedFormat]:
        """Detect file format by magic bytes/signature"""
        try:
            if isinstance(file_data, bytes):
                header = file_data[:8]
            else:
                current_pos = file_data.tell()
                file_data.seek(0)
                header = file_data.read(8)
                file_data.seek(current_pos)
            
            # PDF signature
            if header.startswith(b'%PDF'):
                return SupportedFormat.PDF
            
            # DOC signature (OLE compound document)
            if header.startswith(b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'):
                return SupportedFormat.DOC
            
            # DOCX signature (ZIP archive)
            if header.startswith(b'PK\x03\x04') or header.startswith(b'PK\x05\x06') or header.startswith(b'PK\x07\x08'):
                return SupportedFormat.DOCX
            
            # RTF signature
            if header.startswith(b'{\\rtf'):
                return SupportedFormat.RTF
            
            # Plain text (no specific signature, but check if it's readable text)
            try:
                if isinstance(file_data, bytes):
                    file_data[:100].decode('utf-8')
                else:
                    current_pos = file_data.tell()
                    file_data.seek(0)
                    file_data.read(100).decode('utf-8')
                    file_data.seek(current_pos)
                return SupportedFormat.TXT
            except UnicodeDecodeError:
                pass
            
        except Exception as e:
            logger.debug(f"Error detecting file signature: {e}")
        
        return None
    
    async def _extract_text_from_file(
        self, 
        file_data: Union[bytes, BinaryIO], 
        file_format: SupportedFormat, 
        filename: str
    ) -> str:
        """Extract text from file based on format"""
        
        if file_format == SupportedFormat.TXT:
            return await self._extract_text_from_txt(file_data)
        elif file_format == SupportedFormat.PDF:
            return await self._extract_text_from_pdf(file_data)
        elif file_format == SupportedFormat.DOC:
            return await self._extract_text_from_doc(file_data)
        elif file_format == SupportedFormat.DOCX:
            return await self._extract_text_from_docx(file_data)
        elif file_format == SupportedFormat.RTF:
            return await self._extract_text_from_rtf(file_data)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
    
    async def _extract_text_from_txt(self, file_data: Union[bytes, BinaryIO]) -> str:
        """Extract text from plain text file"""
        try:
            if isinstance(file_data, bytes):
                # Try different encodings
                for encoding in ['utf-8', 'utf-16', 'latin-1', 'cp1252']:
                    try:
                        return file_data.decode(encoding)
                    except UnicodeDecodeError:
                        continue
                
                # If all encodings fail, use utf-8 with error handling
                return file_data.decode('utf-8', errors='replace')
            
            else:
                # File-like object
                content = file_data.read()
                if isinstance(content, bytes):
                    return content.decode('utf-8', errors='replace')
                return content
                
        except Exception as e:
            logger.error(f"Error extracting text from TXT file: {e}")
            return ""
    
    async def _extract_text_from_pdf(self, file_data: Union[bytes, BinaryIO]) -> str:
        """Extract text from PDF file"""
        try:
            import PyPDF2
            import io
            
            if isinstance(file_data, bytes):
                pdf_file = io.BytesIO(file_data)
            else:
                pdf_file = file_data
            
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text_content = []
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text_content.append(page.extract_text())
            
            return '\n'.join(text_content)
            
        except ImportError:
            logger.error("PyPDF2 not available for PDF processing")
            return "Error: PDF processing library not available"
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return f"Error extracting PDF content: {str(e)}"
    
    async def _extract_text_from_doc(self, file_data: Union[bytes, BinaryIO]) -> str:
        """Extract text from DOC file"""
        try:
            # For DOC files, we'd typically use python-docx2txt or similar
            # For now, return a placeholder indicating the limitation
            logger.warning("DOC file processing not fully implemented")
            return "DOC file detected - text extraction not fully implemented in this version"
            
        except Exception as e:
            logger.error(f"Error extracting text from DOC: {e}")
            return f"Error extracting DOC content: {str(e)}"
    
    async def _extract_text_from_docx(self, file_data: Union[bytes, BinaryIO]) -> str:
        """Extract text from DOCX file"""
        try:
            import python_docx
            import io
            
            if isinstance(file_data, bytes):
                docx_file = io.BytesIO(file_data)
            else:
                docx_file = file_data
            
            doc = python_docx.Document(docx_file)
            text_content = []
            
            for paragraph in doc.paragraphs:
                text_content.append(paragraph.text)
            
            return '\n'.join(text_content)
            
        except ImportError:
            logger.error("python-docx not available for DOCX processing")
            return "Error: DOCX processing library not available"
        except Exception as e:
            logger.error(f"Error extracting text from DOCX: {e}")
            return f"Error extracting DOCX content: {str(e)}"
    
    async def _extract_text_from_rtf(self, file_data: Union[bytes, BinaryIO]) -> str:
        """Extract text from RTF file"""
        try:
            # Basic RTF text extraction (simplified)
            if isinstance(file_data, bytes):
                rtf_content = file_data.decode('utf-8', errors='replace')
            else:
                rtf_content = file_data.read()
                if isinstance(rtf_content, bytes):
                    rtf_content = rtf_content.decode('utf-8', errors='replace')
            
            # Simple RTF parsing - remove RTF control codes
            import re
            
            # Remove RTF header and control words
            text = re.sub(r'\\[a-z]+\d*\s?', '', rtf_content)
            text = re.sub(r'[{}]', '', text)
            text = re.sub(r'\\\*.*?;', '', text)
            
            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from RTF: {e}")
            return f"Error extracting RTF content: {str(e)}"
    
    def _calculate_extraction_confidence(
        self, 
        extracted_text: str, 
        file_format: SupportedFormat, 
        filename: str
    ) -> float:
        """Calculate confidence score for text extraction"""
        confidence = 0.5  # Base confidence
        
        # Boost confidence based on text length
        if len(extracted_text) > 100:
            confidence += 0.2
        elif len(extracted_text) > 50:
            confidence += 0.1
        
        # Boost confidence for simpler formats
        if file_format == SupportedFormat.TXT:
            confidence += 0.3
        elif file_format == SupportedFormat.PDF:
            confidence += 0.1
        elif file_format == SupportedFormat.DOCX:
            confidence += 0.2
        
        # Reduce confidence if extraction resulted in error messages
        if "error" in extracted_text.lower():
            confidence -= 0.3
        
        # Check if text looks like medical content
        medical_keywords = ['patient', 'diagnosis', 'treatment', 'medication', 'symptom']
        medical_keyword_count = sum(1 for keyword in medical_keywords if keyword in extracted_text.lower())
        
        if medical_keyword_count >= 2:
            confidence += 0.1
        
        return min(max(confidence, 0.0), 1.0)
    
    async def process_and_summarize_document(
        self, 
        file_data: Union[bytes, BinaryIO], 
        filename: str,
        file_size: Optional[int] = None,
        summary_length: str = "detailed"
    ) -> Dict[str, Any]:
        """Process document and generate summary in one step"""
        
        # First, process the document to extract text
        processed_doc = await self.process_uploaded_file(file_data, filename, file_size)
        
        if processed_doc.error:
            return {
                'success': False,
                'error': processed_doc.error,
                'document_info': processed_doc.__dict__
            }
        
        # Then, summarize the extracted text
        try:
            summary_result = await self.document_summarizer.summarize_document(
                processed_doc.extracted_text,
                summary_length=summary_length,
                extract_entities=True
            )
            
            # Combine results
            combined_result = {
                'success': True,
                'document_info': {
                    'filename': processed_doc.filename,
                    'file_size': processed_doc.file_size,
                    'file_type': processed_doc.file_type,
                    'text_length': processed_doc.text_length,
                    'extraction_confidence': processed_doc.confidence_score,
                    'processing_time': processed_doc.processing_time
                },
                'summary_info': summary_result,
                'extracted_text': processed_doc.extracted_text if len(processed_doc.extracted_text) < 1000 else processed_doc.extracted_text[:1000] + "...",
                'metadata': {
                    'total_processing_time': processed_doc.processing_time + summary_result.get('processing_time', 0),
                    'extraction_method': f"{processed_doc.file_type}_extraction",
                    'summary_method': summary_result.get('method_used', 'unknown')
                }
            }
            
            return combined_result
            
        except Exception as e:
            logger.error(f"Error summarizing document: {e}")
            return {
                'success': False,
                'error': f"Document processed but summarization failed: {str(e)}",
                'document_info': processed_doc.__dict__,
                'extracted_text': processed_doc.extracted_text[:500] + "..." if len(processed_doc.extracted_text) > 500 else processed_doc.extracted_text
            }
    
    def get_supported_formats(self) -> Dict[str, Any]:
        """Get information about supported file formats"""
        return {
            'supported_extensions': list(self.supported_extensions.keys()),
            'supported_formats': [fmt.value for fmt in SupportedFormat],
            'max_file_size': self.max_file_size,
            'max_file_size_mb': self.max_file_size / (1024 * 1024),
            'format_descriptions': {
                'pdf': 'Portable Document Format - requires PyPDF2',
                'txt': 'Plain text files - fully supported',
                'doc': 'Microsoft Word 97-2003 - limited support',
                'docx': 'Microsoft Word 2007+ - requires python-docx',
                'rtf': 'Rich Text Format - basic support'
            }
        }
    
    def validate_file_before_upload(self, filename: str, file_size: int) -> Dict[str, Any]:
        """Validate file before processing (for client-side checks)"""
        return self._validate_file(filename, file_size)
    
    async def extract_text_only(
        self, 
        file_data: Union[bytes, BinaryIO], 
        filename: str
    ) -> Dict[str, Any]:
        """Extract text only without summarization"""
        processed_doc = await self.process_uploaded_file(file_data, filename)
        
        return {
            'success': not bool(processed_doc.error),
            'extracted_text': processed_doc.extracted_text,
            'metadata': {
                'filename': processed_doc.filename,
                'file_type': processed_doc.file_type,
                'text_length': processed_doc.text_length,
                'confidence_score': processed_doc.confidence_score,
                'processing_time': processed_doc.processing_time,
                'word_count': processed_doc.metadata.get('word_count', 0) if processed_doc.metadata else 0
            },
            'error': processed_doc.error
        }