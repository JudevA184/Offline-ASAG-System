import os
import time
import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from contextlib import contextmanager
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from datetime import datetime, timedelta
import sqlite3
from collections import defaultdict
import pickle

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import pdfplumber
from transformers import BartTokenizer, pipeline
from werkzeug.utils import secure_filename
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping
from reportlab.lib.utils import simpleSplit
import joblib
import redis
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from extractor import pdf_to_images, detect_tables, extract_table_text

# Configuration
@dataclass
class Config:
    UPLOAD_FOLDER: str = "uploads"
    OUTPUT_FOLDER: str = "output"
    CACHE_FOLDER: str = "cache"
    HISTORY_FOLDER: str = "history"
    DATABASE_PATH: str = "document_history.db"
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    MAX_FILES_PER_REQUEST: int = 10
    ALLOWED_EXTENSIONS: set = field(default_factory=lambda: {'.pdf'})
    CONFIDENCE_THRESHOLD: float = 0.6
    MAX_TOKENS: int = 512
    SUMMARIZATION_MAX_LENGTH: int = 400
    SUMMARIZATION_MIN_LENGTH: int = 40
    REDIS_URL: str = "redis://localhost:6379"
    LOG_LEVEL: str = "INFO"
    WORKERS: int = 4
    SIMILARITY_THRESHOLD: float = 0.3  # Minimum similarity to consider documents related
    HISTORY_RETENTION_DAYS: int = 365  # Keep history for 1 year

config = Config()

# Enhanced Logging Setup
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Error Classes
class PDFProcessingError(Exception):
    pass

class ClassificationError(Exception):
    pass

class SummarizationError(Exception):
    pass

class HistoryAnalysisError(Exception):
    pass

# Document Metadata
@dataclass
class DocumentMetadata:
    id: str
    filename: str
    upload_date: datetime
    file_hash: str
    categories: Dict[str, int]  # Category -> paragraph count
    key_metrics: Dict[str, float]  # Extracted numerical data
    content_summary: str
    text_length: int
    company_name: Optional[str] = None
    document_type: Optional[str] = None
    time_period: Optional[str] = None

# Historical Analysis Engine
class HistoricalAnalyzer:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database for document history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                upload_date TIMESTAMP NOT NULL,
                file_hash TEXT UNIQUE NOT NULL,
                categories TEXT NOT NULL,
                key_metrics TEXT NOT NULL,
                content_summary TEXT NOT NULL,
                text_length INTEGER NOT NULL,
                company_name TEXT,
                document_type TEXT,
                time_period TEXT,
                content_vector BLOB
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_relationships (
                doc1_id TEXT NOT NULL,
                doc2_id TEXT NOT NULL,
                similarity_score REAL NOT NULL,
                relationship_type TEXT NOT NULL,
                created_date TIMESTAMP NOT NULL,
                PRIMARY KEY (doc1_id, doc2_id),
                FOREIGN KEY (doc1_id) REFERENCES documents (id),
                FOREIGN KEY (doc2_id) REFERENCES documents (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def extract_key_metrics(self, text: str, categories: Dict[str, List[str]]) -> Dict[str, float]:
        """Extract numerical metrics from document text"""
        metrics = {}
        
        # Financial metrics patterns
        financial_patterns = {
            'revenue': r'revenue[:\s]*\$?([\d,]+\.?\d*)\s*(?:million|billion|k|m|b)?',
            'profit': r'(?:profit|income)[:\s]*\$?([\d,]+\.?\d*)\s*(?:million|billion|k|m|b)?',
            'loss': r'loss[:\s]*\$?([\d,]+\.?\d*)\s*(?:million|billion|k|m|b)?',
            'growth': r'growth[:\s]*([\d,]+\.?\d*)%?',
            'percentage': r'([\d,]+\.?\d*)%',
            'sales': r'sales[:\s]*\$?([\d,]+\.?\d*)\s*(?:million|billion|k|m|b)?',
            'expenses': r'expenses?[:\s]*\$?([\d,]+\.?\d*)\s*(?:million|billion|k|m|b)?',
            'budget': r'budget[:\s]*\$?([\d,]+\.?\d*)\s*(?:million|billion|k|m|b)?',
        }
        
        text_lower = text.lower()
        
        for metric_name, pattern in financial_patterns.items():
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                # Convert to float and store the largest value found
                values = []
                for match in matches:
                    try:
                        # Remove commas and convert to float
                        value = float(match.replace(',', ''))
                        values.append(value)
                    except ValueError:
                        continue
                
                if values:
                    metrics[metric_name] = max(values)
        
        # Extract years/dates
        year_pattern = r'\b(19|20)\d{2}\b'
        years = re.findall(year_pattern, text)
        if years:
            metrics['years_mentioned'] = len(set(years))
            metrics['latest_year'] = max([int(y) for y in years])
        
        # Count specific category mentions
        for category, paragraphs in categories.items():
            metrics[f'{category.lower()}_mentions'] = len(paragraphs)
        
        return metrics
    
    def detect_document_type(self, text: str, filename: str) -> str:
        """Detect document type based on content and filename"""
        text_lower = text.lower()
        filename_lower = filename.lower()
        
        # Define document type patterns
        type_patterns = {
            'financial_report': ['financial', 'annual report', 'quarterly', 'earnings', 'balance sheet'],
            'sales_report': ['sales', 'revenue', 'marketing', 'customer'],
            'risk_assessment': ['risk', 'assessment', 'threat', 'vulnerability'],
            'legal_document': ['legal', 'contract', 'agreement', 'litigation'],
            'investment_report': ['investment', 'portfolio', 'asset', 'fund'],
            'operational_report': ['operations', 'operational', 'performance', 'efficiency'],
            'compliance_report': ['compliance', 'regulatory', 'audit', 'governance']
        }
        
        # Check filename first
        for doc_type, keywords in type_patterns.items():
            if any(keyword in filename_lower for keyword in keywords):
                return doc_type
        
        # Check content
        for doc_type, keywords in type_patterns.items():
            keyword_count = sum(1 for keyword in keywords if keyword in text_lower)
            if keyword_count >= 2:  # At least 2 keywords match
                return doc_type
        
        return 'general'
    
    def extract_time_period(self, text: str, filename: str) -> Optional[str]:
        """Extract time period from document"""
        # Check filename first
        filename_lower = filename.lower()
        
        # Year patterns
        year_match = re.search(r'\b(19|20)\d{2}\b', filename_lower)
        if year_match:
            return year_match.group(0)
        
        # Quarter patterns
        quarter_match = re.search(r'q[1-4]\s*(19|20)\d{2}', filename_lower)
        if quarter_match:
            return quarter_match.group(0)
        
        # Month patterns
        month_match = re.search(r'(january|february|march|april|may|june|july|august|september|october|november|december)\s*(19|20)\d{2}', filename_lower)
        if month_match:
            return month_match.group(0)
        
        # Check content for time periods
        text_lower = text.lower()
        
        # Look for "for the year" patterns
        year_pattern = r'for the year\s*(19|20)\d{2}'
        year_match = re.search(year_pattern, text_lower)
        if year_match:
            return year_match.group(1)
        
        # Look for quarterly patterns
        quarter_pattern = r'(first|second|third|fourth|q[1-4])\s*quarter\s*(19|20)\d{2}'
        quarter_match = re.search(quarter_pattern, text_lower)
        if quarter_match:
            return f"{quarter_match.group(1)} quarter {quarter_match.group(2)}"
        
        return None
    
    def extract_company_name(self, text: str, filename: str) -> Optional[str]:
        """Extract company name from document"""
        # Common company patterns
        company_patterns = [
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:Inc\.|LLC|Corp\.|Corporation|Ltd\.)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:Company|Group|Industries)',
            r'(?:Company|Corporation):\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        ]
        
        for pattern in company_patterns:
            matches = re.findall(pattern, text[:1000])  # Check first 1000 chars
            if matches:
                return matches[0].strip()
        
        # Check filename for company name
        filename_parts = filename.replace('_', ' ').replace('-', ' ').split()
        for part in filename_parts:
            if part.lower() in ['inc', 'corp', 'ltd', 'llc']:
                idx = filename_parts.index(part)
                if idx > 0:
                    return ' '.join(filename_parts[:idx+1])
        
        return None
    
    def store_document(self, metadata: DocumentMetadata, content_vector: np.ndarray):
        """Store document metadata and vector in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO documents 
                (id, filename, upload_date, file_hash, categories, key_metrics, 
                 content_summary, text_length, company_name, document_type, 
                 time_period, content_vector)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metadata.id,
                metadata.filename,
                metadata.upload_date,
                metadata.file_hash,
                json.dumps(metadata.categories),
                json.dumps(metadata.key_metrics),
                metadata.content_summary,
                metadata.text_length,
                metadata.company_name,
                metadata.document_type,
                metadata.time_period,
                pickle.dumps(content_vector)
            ))
            
            conn.commit()
            logger.info(f"Stored document metadata: {metadata.id}")
            
        except sqlite3.Error as e:
            logger.error(f"Database error storing document: {e}")
            raise HistoryAnalysisError(f"Failed to store document: {e}")
        finally:
            conn.close()
    
    def find_related_documents(self, current_doc: DocumentMetadata, 
                             content_vector: np.ndarray) -> List[Tuple[DocumentMetadata, float, str]]:
        """Find documents related to the current document"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get all documents from the same company or similar type
            cursor.execute('''
                SELECT * FROM documents 
                WHERE id != ? 
                AND (company_name = ? OR document_type = ?)
                ORDER BY upload_date DESC
                LIMIT 50
            ''', (current_doc.id, current_doc.company_name, current_doc.document_type))
            
            rows = cursor.fetchall()
            related_docs = []
            
            for row in rows:
                # Reconstruct metadata
                doc_metadata = DocumentMetadata(
                    id=row[0],
                    filename=row[1],
                    upload_date=datetime.fromisoformat(row[2]),
                    file_hash=row[3],
                    categories=json.loads(row[4]),
                    key_metrics=json.loads(row[5]),
                    content_summary=row[6],
                    text_length=row[7],
                    company_name=row[8],
                    document_type=row[9],
                    time_period=row[10]
                )
                
                # Calculate similarity
                stored_vector = pickle.loads(row[11])
                similarity = cosine_similarity([content_vector], [stored_vector])[0][0]
                
                if similarity >= config.SIMILARITY_THRESHOLD:
                    relationship_type = self._determine_relationship_type(current_doc, doc_metadata)
                    related_docs.append((doc_metadata, similarity, relationship_type))
            
            # Sort by similarity
            related_docs.sort(key=lambda x: x[1], reverse=True)
            return related_docs[:10]  # Return top 10 most similar
            
        except sqlite3.Error as e:
            logger.error(f"Database error finding related documents: {e}")
            return []
        finally:
            conn.close()
    
    def _determine_relationship_type(self, current_doc: DocumentMetadata, 
                                   related_doc: DocumentMetadata) -> str:
        """Determine the type of relationship between documents"""
        if current_doc.company_name == related_doc.company_name:
            if current_doc.document_type == related_doc.document_type:
                # Same company, same type - likely temporal relationship
                if current_doc.time_period and related_doc.time_period:
                    try:
                        current_year = int(re.search(r'\d{4}', current_doc.time_period).group())
                        related_year = int(re.search(r'\d{4}', related_doc.time_period).group())
                        if current_year > related_year:
                            return 'temporal_successor'
                        elif current_year < related_year:
                            return 'temporal_predecessor'
                    except:
                        pass
                return 'same_type_same_company'
            else:
                return 'same_company_different_type'
        elif current_doc.document_type == related_doc.document_type:
            return 'same_type_different_company'
        else:
            return 'content_similar'
    
    def generate_historical_insights(self, current_doc: DocumentMetadata, 
                                   related_docs: List[Tuple[DocumentMetadata, float, str]]) -> str:
        """Generate insights based on historical document analysis"""
        if not related_docs:
            return "No related historical documents found for comparison."
        
        insights = []
        insights.append("## Historical Analysis ##")
        
        # Group by relationship type
        relationship_groups = defaultdict(list)
        for doc, similarity, rel_type in related_docs:
            relationship_groups[rel_type].append((doc, similarity))
        
        # Temporal analysis
        if 'temporal_successor' in relationship_groups or 'temporal_predecessor' in relationship_groups:
            temporal_insights = self._analyze_temporal_trends(current_doc, relationship_groups)
            if temporal_insights:
                insights.append("### Temporal Trends ###")
                insights.extend(temporal_insights)
        
        # Metric comparisons
        metric_insights = self._compare_metrics(current_doc, related_docs)
        if metric_insights:
            insights.append("### Key Metrics Comparison ###")
            insights.extend(metric_insights)
        
        # Content evolution
        content_insights = self._analyze_content_evolution(current_doc, related_docs)
        if content_insights:
            insights.append("### Content Evolution ###")
            insights.extend(content_insights)
        
        # Document pattern analysis
        pattern_insights = self._analyze_document_patterns(current_doc, related_docs)
        if pattern_insights:
            insights.append("### Document Patterns ###")
            insights.extend(pattern_insights)
        
        return '\n\n'.join(insights) if len(insights) > 1 else "Limited historical data available for detailed analysis."
    
    def _analyze_temporal_trends(self, current_doc: DocumentMetadata, 
                               relationship_groups: Dict) -> List[str]:
        """Analyze temporal trends in document metrics"""
        insights = []
        
        # Find predecessor documents
        predecessors = relationship_groups.get('temporal_predecessor', [])
        if not predecessors:
            return insights
        
        # Sort by time period
        sorted_predecessors = sorted(predecessors, key=lambda x: x[0].time_period or "")
        
        # Compare key metrics
        for metric, current_value in current_doc.key_metrics.items():
            if metric in ['revenue', 'sales', 'profit', 'growth']:
                prev_values = []
                for pred_doc, _ in sorted_predecessors:
                    if metric in pred_doc.key_metrics:
                        prev_values.append((pred_doc.time_period, pred_doc.key_metrics[metric]))
                
                if prev_values:
                    latest_prev = prev_values[-1]
                    if latest_prev[1] > 0:  # Avoid division by zero
                        change_pct = ((current_value - latest_prev[1]) / latest_prev[1]) * 100
                        trend = "increased" if change_pct > 0 else "decreased"
                        insights.append(f"{metric.title()} has {trend} by {abs(change_pct):.1f}% since {latest_prev[0]}")
        
        return insights
    
    def _compare_metrics(self, current_doc: DocumentMetadata, 
                        related_docs: List[Tuple[DocumentMetadata, float, str]]) -> List[str]:
        """Compare key metrics across related documents"""
        insights = []
        
        # Aggregate metrics from related documents
        metric_comparisons = defaultdict(list)
        
        for doc, similarity, rel_type in related_docs:
            for metric, value in doc.key_metrics.items():
                metric_comparisons[metric].append((value, doc.time_period, rel_type))
        
        # Generate insights for each metric
        for metric, current_value in current_doc.key_metrics.items():
            if metric in metric_comparisons:
                related_values = metric_comparisons[metric]
                
                if len(related_values) >= 2:
                    values = [v[0] for v in related_values]
                    avg_value = sum(values) / len(values)
                    
                    if avg_value > 0:
                        diff_pct = ((current_value - avg_value) / avg_value) * 100
                        comparison = "above" if diff_pct > 0 else "below"
                        insights.append(f"Current {metric} is {abs(diff_pct):.1f}% {comparison} historical average")
        
        return insights
    
    def _analyze_content_evolution(self, current_doc: DocumentMetadata, 
                                 related_docs: List[Tuple[DocumentMetadata, float, str]]) -> List[str]:
        """Analyze how document content has evolved"""
        insights = []
        
        # Compare category distributions
        current_categories = current_doc.categories
        
        # Find documents of the same type
        same_type_docs = [doc for doc, _, rel_type in related_docs 
                         if rel_type in ['same_type_same_company', 'temporal_predecessor']]
        
        if same_type_docs:
            # Analyze category focus changes
            category_changes = {}
            for doc in same_type_docs:
                for category, count in doc.categories.items():
                    if category in current_categories:
                        current_count = current_categories[category]
                        change = current_count - count
                        if category not in category_changes:
                            category_changes[category] = []
                        category_changes[category].append(change)
            
            # Summarize significant changes
            for category, changes in category_changes.items():
                if changes:
                    avg_change = sum(changes) / len(changes)
                    if abs(avg_change) > 1:  # Significant change threshold
                        direction = "increased" if avg_change > 0 else "decreased"
                        insights.append(f"Focus on {category} has {direction} compared to historical documents")
        
        return insights
    
    def _analyze_document_patterns(self, current_doc: DocumentMetadata, 
                                 related_docs: List[Tuple[DocumentMetadata, float, str]]) -> List[str]:
        """Analyze patterns in document structure and content"""
        insights = []
        
        # Document length analysis
        related_lengths = [doc.text_length for doc, _, _ in related_docs]
        if related_lengths:
            avg_length = sum(related_lengths) / len(related_lengths)
            length_diff_pct = ((current_doc.text_length - avg_length) / avg_length) * 100
            
            if abs(length_diff_pct) > 20:  # Significant difference
                comparison = "longer" if length_diff_pct > 0 else "shorter"
                insights.append(f"Document is {abs(length_diff_pct):.0f}% {comparison} than similar historical documents")
        
        # Frequency analysis
        upload_dates = [doc.upload_date for doc, _, _ in related_docs]
        if len(upload_dates) >= 2:
            # Calculate average time between uploads
            sorted_dates = sorted(upload_dates)
            intervals = [(sorted_dates[i+1] - sorted_dates[i]).days for i in range(len(sorted_dates)-1)]
            avg_interval = sum(intervals) / len(intervals)
            
            # Check if current upload follows pattern
            last_upload = max(upload_dates)
            current_interval = (current_doc.upload_date - last_upload).days
            
            if abs(current_interval - avg_interval) > 30:  # 30 days threshold
                if current_interval > avg_interval:
                    insights.append("Upload timing is later than usual historical pattern")
                else:
                    insights.append("Upload timing is earlier than usual historical pattern")
        
        return insights

# Enhanced Text Processing (keeping existing functionality)
class TextProcessor:
    @staticmethod
    def clean_extracted_text(text: str) -> str:
        """Enhanced text cleaning with better regex patterns"""
        if not text:
            return ""
        
        # Remove excessive newlines and replace with periods
        text = re.sub(r'\n+', '. ', text)
        text = re.sub(r'(?<=\w)\n(?=\w)', ' ', text)
        
        # Clean up spacing
        text = re.sub(r'\s{2,}', ' ', text)
        text = re.sub(r'\. \.+', '.', text)
        text = re.sub(r'(?<=\d)\s+(?=\d)', '', text)
        
        # Better word-number separation
        text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
        text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
        
        # Remove standalone periods
        text = re.sub(r'(?<!\w)\.(?!\w)', '', text)
        
        # Fix common OCR errors
        text = re.sub(r'\b0\b', 'O', text)  # Zero to O
        text = re.sub(r'\bl\b', 'I', text)  # lowercase l to I
        
        return text.strip()

    @staticmethod
    def split_into_paragraphs(text: str) -> List[str]:
        """Better paragraph splitting"""
        # Split on double newlines or sentence boundaries with capitals
        paragraphs = re.split(r'\n\s*\n|(?<=\.)\s{2,}(?=[A-Z])', text)
        return [p.strip() for p in paragraphs if len(p.strip()) >= 30]

# Enhanced Classification (keeping existing functionality)
class DocumentClassifier:
    def __init__(self, model_path: str, confidence_threshold: float = 0.6):
        self.classifier = joblib.load(model_path)
        self.confidence_threshold = confidence_threshold
        self.categories = {
            "Risks": [],
            "Financials": [],
            "Litigations": [],
            "Investments": [],
            "General": []
        }
        
        # Keyword mapping for fallback classification
        self.keyword_mapping = {
            "Risks": ["risk", "threat", "vulnerability", "danger", "hazard", "uncertainty"],
            "Financials": ["revenue", "profit", "income", "financial", "earnings", "budget", "cost"],
            "Litigations": ["legal", "court", "lawsuit", "litigation", "dispute", "settlement"],
            "Investments": ["investment", "portfolio", "asset", "equity", "bond", "fund", "capital"]
        }

    def classify_paragraphs(self, text: str) -> Dict[str, List[str]]:
        """Enhanced classification with better error handling"""
        categories = {key: [] for key in self.categories.keys()}
        paragraphs = TextProcessor.split_into_paragraphs(text)
        
        for para in paragraphs:
            try:
                category = self._classify_single_paragraph(para)
                categories[category].append(para)
            except Exception as e:
                logger.warning(f"Classification error for paragraph: {str(e)}")
                categories["General"].append(para)
        
        return categories

    def _classify_single_paragraph(self, paragraph: str) -> str:
        """Classify a single paragraph with confidence scoring"""
        try:
            pred_proba = self.classifier.predict_proba([paragraph])[0]
            max_conf = max(pred_proba)
            predicted = self.classifier.classes_[pred_proba.argmax()]
            
            if max_conf >= self.confidence_threshold:
                return predicted
            else:
                return self._fallback_classification(paragraph)
                
        except Exception as e:
            logger.error(f"Classifier prediction error: {str(e)}")
            return self._fallback_classification(paragraph)

    def _fallback_classification(self, paragraph: str) -> str:
        """Keyword-based fallback classification"""
        para_lower = paragraph.lower()
        
        for category, keywords in self.keyword_mapping.items():
            if any(keyword in para_lower for keyword in keywords):
                return category
        
        return "General"

# Enhanced Summarization (keeping existing functionality)
class DocumentSummarizer:
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.summarizer = pipeline("summarization", model=model_name)
        self.max_tokens = config.MAX_TOKENS
        
    def chunk_text(self, text: str) -> List[str]:
        """Improved text chunking with overlap"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_len = 0
        overlap_size = 50  # Words to overlap between chunks
        
        for word in words:
            token_len = len(self.tokenizer.tokenize(word))
            
            if current_len + token_len > self.max_tokens:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    # Keep last overlap_size words for next chunk
                    current_chunk = current_chunk[-overlap_size:] if len(current_chunk) > overlap_size else []
                    current_len = sum(len(self.tokenizer.tokenize(w)) for w in current_chunk)
                    
            current_chunk.append(word)
            current_len += token_len
            
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

    def summarize_by_category(self, classified_text: Dict[str, List[str]]) -> str:
        """Enhanced summarization with better error handling"""
        summarized_sections = []
        
        for section, paragraphs in classified_text.items():
            if not paragraphs:
                continue
                
            try:
                section_summary = self._summarize_section(section, paragraphs)
                if section_summary:
                    summarized_sections.append(f"### {section} ###\n{section_summary}")
            except Exception as e:
                logger.error(f"Error summarizing {section}: {str(e)}")
                # Add a fallback summary
                sample_text = paragraphs[0][:200] + "..." if paragraphs else "No content available"
                summarized_sections.append(f"### {section} ###\n[Summary unavailable] {sample_text}")
        
        return "\n\n".join(summarized_sections)

    def _summarize_section(self, section: str, paragraphs: List[str]) -> str:
        """Summarize a single section"""
        section_text = TextProcessor.clean_extracted_text(" ".join(paragraphs))
        
        if len(section_text) < 100:  # Too short to summarize
            return section_text
            
        chunks = self.chunk_text(section_text)
        summaries = []
        
        for i, chunk in enumerate(chunks):
            try:
                output = self.summarizer(
                    chunk, 
                    max_length=config.SUMMARIZATION_MAX_LENGTH,
                    min_length=config.SUMMARIZATION_MIN_LENGTH,
                    do_sample=False
                )
                summaries.append(output[0]['summary_text'])
            except Exception as e:
                logger.warning(f"Error summarizing chunk {i} in {section}: {str(e)}")
                # Use first 200 chars as fallback
                summaries.append(chunk[:200] + "...")
        
        return " ".join(summaries)

# Enhanced PDF Processing (keeping existing functionality)
class PDFProcessor:
    @staticmethod
    def extract_text_and_tables(pdf_path: str) -> str:
        """Enhanced PDF extraction with better error handling"""
        full_text = ""
        
        try:
            # Extract text using pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    try:
                        text = page.extract_text()
                        if text:
                            full_text += f"\n[Page {i+1} Text]\n{text.strip()}\n"
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {i+1}: {str(e)}")
                        
        except Exception as e:
            logger.error(f"Error opening PDF with pdfplumber: {str(e)}")
            raise PDFProcessingError(f"Failed to process PDF: {str(e)}")
        
        # Extract tables using computer vision
        try:
            images = pdf_to_images(pdf_path)
            for i, image in enumerate(images):
                try:
                    results = detect_tables(image)
                    boxes = [
                        box.int().tolist() 
                        for score, label, box in zip(results["scores"], results["labels"], results["boxes"])
                        if label.item() == 1 and score.item() > 0.9
                    ]
                    
                    table_texts = extract_table_text(image, boxes)
                    for j, table_text in enumerate(table_texts):
                        full_text += f"\n[Page {i+1} Table {j+1} OCR]\n{table_text}\n"
                        
                except Exception as e:
                    logger.warning(f"Error processing tables on page {i+1}: {str(e)}")
                    
        except Exception as e:
            logger.warning(f"Error in table extraction: {str(e)}")
        
        return full_text.strip()

    @staticmethod
    def generate_summary_pdf(summary_text: str, historical_insights: str, output_path: str) -> None:
        """Enhanced PDF generation with historical insights"""
        try:
            # Register font with fallback
            try:
                pdfmetrics.registerFont(TTFont("DejaVu", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
            except:
                try:
                    pdfmetrics.registerFont(TTFont("DejaVu", "DejaVuSans.ttf"))
                except:
                    logger.warning("Could not load DejaVu font, using default")
                    
            c = canvas.Canvas(output_path, pagesize=letter)
            width, height = letter
            margin = 50
            y = height - 70
            
            # Title
            c.setFont("Helvetica-Bold", 16)
            c.drawCentredString(width / 2, y, "Smart Document Analysis Report")
            c.drawCentredString(width / 2, y - 20, f"Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}")
            y -= 60
            
            # Content
            c.setFont("Helvetica", 10)
            line_height = 14
            max_width = width - 2 * margin
            
            # Combine summary and historical insights
            full_content = summary_text
            if historical_insights:
                full_content += f"\n\n{historical_insights}"
            
            lines = []
            for line in full_content.split('\n'):
                if not line.strip():
                    lines.append('')
                else:
                    # Handle section headers
                    if line.startswith('###') or line.startswith('##'):
                        lines.append('')  # Add space before section
                        lines.append(line.replace('#', '').strip())
                        lines.append('-' * 40)  # Add underline
                    else:
                        wrapped = simpleSplit(line, "Helvetica", 10, max_width)
                        lines.extend(wrapped)
            
            for line in lines:
                if y < margin + line_height:
                    c.showPage()
                    y = height - margin
                    c.setFont("Helvetica", 10)
                    
                # Bold section headers
                if line and not line.startswith(' ') and not line.startswith('-') and len(line) < 50:
                    c.setFont("Helvetica-Bold", 11)
                    c.drawString(margin, y, line)
                    c.setFont("Helvetica", 10)
                else:
                    c.drawString(margin, y, line)
                    
                y -= line_height
                
            c.save()
            logger.info(f"Successfully generated PDF: {output_path}")
            
        except Exception as e:
            logger.error(f"Error generating PDF: {str(e)}")
            raise

# Caching System (keeping existing functionality)
class CacheManager:
    def __init__(self, cache_folder: str):
        self.cache_folder = cache_folder
        os.makedirs(cache_folder, exist_ok=True)
        
    def get_file_hash(self, file_path: str) -> str:
        """Generate hash for file caching"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def get_cached_result(self, file_hash: str) -> Optional[str]:
        """Get cached processing result"""
        cache_path = os.path.join(self.cache_folder, f"{file_hash}.txt")
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                return f.read()
        return None
    
    def cache_result(self, file_hash: str, result: str) -> None:
        """Cache processing result"""
        cache_path = os.path.join(self.cache_folder, f"{file_hash}.txt")
        with open(cache_path, 'w', encoding='utf-8') as f:
            f.write(result)

# Enhanced Flask App
app = Flask(__name__)
CORS(app)

# Rate limiting
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Initialize components
for folder in [config.UPLOAD_FOLDER, config.OUTPUT_FOLDER, config.CACHE_FOLDER, config.HISTORY_FOLDER]:
    os.makedirs(folder, exist_ok=True)

classifier = DocumentClassifier("paragraph_classifier.joblib", config.CONFIDENCE_THRESHOLD)
summarizer = DocumentSummarizer()
cache_manager = CacheManager(config.CACHE_FOLDER)
historical_analyzer = HistoricalAnalyzer(config.DATABASE_PATH)
executor = ThreadPoolExecutor(max_workers=config.WORKERS)

def validate_file(file) -> Tuple[bool, str]:
    """Validate uploaded file"""
    if not file or not file.filename:
        return False, "No file provided"
    
    filename = secure_filename(file.filename)
    if not filename:
        return False, "Invalid filename"
    
    ext = os.path.splitext(filename)[1].lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        return False, f"File type {ext} not allowed"
    
    # Check file size (this is approximate)
    file.seek(0, 2)  # Seek to end
    size = file.tell()
    file.seek(0)  # Reset
    
    if size > config.MAX_FILE_SIZE:
        return False, f"File too large. Max size: {config.MAX_FILE_SIZE // (1024*1024)}MB"
    
    return True, ""

def process_single_file(file_data: Tuple[str, str]) -> Dict:
    """Enhanced process_single_file with historical analysis"""
    upload_path, filename = file_data
    
    try:
        # Check cache first
        file_hash = cache_manager.get_file_hash(upload_path)
        cached_result = cache_manager.get_cached_result(file_hash)
        
        if cached_result:
            logger.info(f"Using cached result for {filename}")
            summary_text = cached_result
            historical_insights = ""  # Skip historical analysis for cached results
        else:
            # Process file
            logger.info(f"Processing {filename}")
            text = PDFProcessor.extract_text_and_tables(upload_path)
            
            if not text.strip():
                summary_text = "No text found in document."
                historical_insights = ""
            else:
                # Classify and summarize
                classified = classifier.classify_paragraphs(text)
                summary_text = summarizer.summarize_by_category(classified)
                
                # Extract metadata for historical analysis
                doc_id = hashlib.sha256(f"{filename}_{time.time()}".encode()).hexdigest()[:16]
                
                # Create content vector for similarity analysis
                try:
                    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
                    content_vector = vectorizer.fit_transform([text]).toarray()[0]
                except Exception as e:
                    logger.warning(f"Error creating content vector: {e}")
                    content_vector = np.zeros(1000)  # Fallback empty vector
                
                # Extract metadata
                metadata = DocumentMetadata(
                    id=doc_id,
                    filename=filename,
                    upload_date=datetime.now(),
                    file_hash=file_hash,
                    categories={k: len(v) for k, v in classified.items()},
                    key_metrics=historical_analyzer.extract_key_metrics(text, classified),
                    content_summary=summary_text[:500],  # First 500 chars
                    text_length=len(text),
                    company_name=historical_analyzer.extract_company_name(text, filename),
                    document_type=historical_analyzer.detect_document_type(text, filename),
                    time_period=historical_analyzer.extract_time_period(text, filename)
                )
                
                # Find related documents and generate insights
                try:
                    related_docs = historical_analyzer.find_related_documents(metadata, content_vector)
                    historical_insights = historical_analyzer.generate_historical_insights(metadata, related_docs)
                    
                    # Store current document for future comparisons
                    historical_analyzer.store_document(metadata, content_vector)
                    
                except Exception as e:
                    logger.error(f"Error in historical analysis: {e}")
                    historical_insights = "Historical analysis unavailable due to processing error."
            
            # Cache result (summary only, not historical insights)
            cache_manager.cache_result(file_hash, summary_text)
        
        # Generate output PDF with historical insights
        timestamp = int(time.time())
        base_name = os.path.splitext(filename)[0]
        summary_pdf_name = f"{base_name}_smart_summary_{timestamp}.pdf"
        summary_pdf_path = os.path.join(config.OUTPUT_FOLDER, summary_pdf_name)
        
        PDFProcessor.generate_summary_pdf(summary_text, historical_insights, summary_pdf_path)
        
        return {
            "filename": filename,
            "summary": summary_text,
            "historical_insights": historical_insights,
            "summary_pdf": f"/download/{summary_pdf_name}",
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error processing {filename}: {str(e)}")
        return {
            "filename": filename,
            "error": str(e),
            "status": "error"
        }
    finally:
        # Cleanup uploaded file
        if os.path.exists(upload_path):
            os.remove(upload_path)

@app.route('/upload', methods=['POST'])
@limiter.limit("10 per minute")
def upload_files():
    """Enhanced file upload endpoint with historical analysis"""
    if 'files' not in request.files:
        return jsonify({"error": "No files part"}), 400
    
    files = request.files.getlist('files')
    if not files or all(not f.filename for f in files):
        return jsonify({"error": "No files uploaded"}), 400
    
    if len(files) > config.MAX_FILES_PER_REQUEST:
        return jsonify({"error": f"Too many files. Max: {config.MAX_FILES_PER_REQUEST}"}), 400
    
    # Validate and save files
    file_data = []
    for file in files:
        is_valid, error_msg = validate_file(file)
        if not is_valid:
            return jsonify({"error": error_msg}), 400
        
        filename = secure_filename(file.filename)
        upload_path = os.path.join(config.UPLOAD_FOLDER, f"{int(time.time())}_{filename}")
        file.save(upload_path)
        file_data.append((upload_path, filename))
    
    # Process files concurrently
    try:
        futures = [executor.submit(process_single_file, data) for data in file_data]
        responses = [future.result() for future in futures]
        
        return jsonify({
            "results": responses,
            "processed": len(responses),
            "successful": len([r for r in responses if r.get("status") == "success"]),
            "message": "Documents processed with historical analysis"
        })
        
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        return jsonify({"error": "Internal processing error"}), 500

@app.route('/download/<path:filename>')
def download_file(filename):
    """Secure file download"""
    try:
        return send_from_directory(config.OUTPUT_FOLDER, filename, as_attachment=True)
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404

@app.route('/history')
def get_document_history():
    """Get document processing history"""
    try:
        conn = sqlite3.connect(config.DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT filename, upload_date, document_type, company_name, time_period
            FROM documents 
            ORDER BY upload_date DESC 
            LIMIT 50
        ''')
        
        rows = cursor.fetchall()
        history = []
        
        for row in rows:
            history.append({
                "filename": row[0],
                "upload_date": row[1],
                "document_type": row[2],
                "company_name": row[3],
                "time_period": row[4]
            })
        
        conn.close()
        return jsonify({"history": history})
        
    except Exception as e:
        logger.error(f"Error retrieving history: {e}")
        return jsonify({"error": "Failed to retrieve history"}), 500

@app.route('/relationships/<doc_id>')
def get_document_relationships(doc_id):
    """Get relationships for a specific document"""
    try:
        conn = sqlite3.connect(config.DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT d2.filename, dr.similarity_score, dr.relationship_type
            FROM document_relationships dr
            JOIN documents d2 ON dr.doc2_id = d2.id
            WHERE dr.doc1_id = ?
            ORDER BY dr.similarity_score DESC
        ''', (doc_id,))
        
        rows = cursor.fetchall()
        relationships = []
        
        for row in rows:
            relationships.append({
                "filename": row[0],
                "similarity_score": row[1],
                "relationship_type": row[2]
            })
        
        conn.close()
        return jsonify({"relationships": relationships})
        
    except Exception as e:
        logger.error(f"Error retrieving relationships: {e}")
        return jsonify({"error": "Failed to retrieve relationships"}), 500

@app.route('/cleanup')
def cleanup_old_data():
    """Clean up old cached data and documents"""
    try:
        cutoff_date = datetime.now() - timedelta(days=config.HISTORY_RETENTION_DAYS)
        
        conn = sqlite3.connect(config.DATABASE_PATH)
        cursor = conn.cursor()
        
        # Delete old documents
        cursor.execute('DELETE FROM documents WHERE upload_date < ?', (cutoff_date,))
        cursor.execute('DELETE FROM document_relationships WHERE created_date < ?', (cutoff_date,))
        
        deleted_docs = cursor.rowcount
        conn.commit()
        conn.close()
        
        # Clean up old cache files
        cache_files_deleted = 0
        if os.path.exists(config.CACHE_FOLDER):
            for filename in os.listdir(config.CACHE_FOLDER):
                file_path = os.path.join(config.CACHE_FOLDER, filename)
                if os.path.isfile(file_path):
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if file_time < cutoff_date:
                        os.remove(file_path)
                        cache_files_deleted += 1
        
        return jsonify({
            "message": "Cleanup completed",
            "documents_deleted": deleted_docs,
            "cache_files_deleted": cache_files_deleted
        })
        
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        return jsonify({"error": "Cleanup failed"}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "version": "3.0-smart",
        "features": ["summarization", "classification", "historical_analysis"]
    })

@app.route('/stats')
def get_stats():
    """Get processing statistics with historical data"""
    try:
        conn = sqlite3.connect(config.DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM documents')
        total_docs = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(DISTINCT company_name) FROM documents WHERE company_name IS NOT NULL')
        unique_companies = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(DISTINCT document_type) FROM documents')
        unique_doc_types = cursor.fetchone()[0]
        
        conn.close()
        
        upload_count = len(os.listdir(config.UPLOAD_FOLDER)) if os.path.exists(config.UPLOAD_FOLDER) else 0
        output_count = len(os.listdir(config.OUTPUT_FOLDER)) if os.path.exists(config.OUTPUT_FOLDER) else 0
        cache_count = len(os.listdir(config.CACHE_FOLDER)) if os.path.exists(config.CACHE_FOLDER) else 0
        
        return jsonify({
            "uploads_processed": upload_count,
            "summaries_generated": output_count,
            "cached_results": cache_count,
            "total_documents_stored": total_docs,
            "unique_companies": unique_companies,
            "unique_document_types": unique_doc_types
        })
        
    except Exception as e:
        logger.error(f"Error retrieving stats: {e}")
        return jsonify({"error": "Failed to retrieve statistics"}), 500

@app.errorhandler(413)
def file_too_large(e):
    return jsonify({"error": "File too large"}), 413

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    logger.info("Starting Smart PDF Summarization Server v3.0")
    logger.info("Features: Document Summarization, Classification, Historical Analysis")
    app.run(host='0.0.0.0', port=5000, debug=False)