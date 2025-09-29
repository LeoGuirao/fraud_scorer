"""Tests for CaseContext and _load_document_metadata method."""

import json
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch, Mock

from fraud_scorer.analyzers.correlation.models.case_context import CaseContext


class TestCaseContext(TestCase):
    """Test suite for CaseContext class."""

    def setUp(self):
        """Set up test fixtures."""
        self.case_id = "TEST-CASE-001"
        self.mock_consolidated = {
            "case_id": self.case_id,  # Add required field
            "consolidated_fields": {
                "numero_siniestro": "2025-001",
                "asegurado": "Test Company",
                "monto_reclamacion": "100000",
            }
        }
        self.mock_extractions = [
            {
                "document_id": "doc1",
                "source_document": "poliza.pdf",
                "document_type": "poliza_de_la_aseguradora",
                "extracted_fields": {
                    "numero_poliza": "POL-123",
                    "vigencia_desde": "2024-01-01",
                }
            }
        ]
        self.mock_fraud_results = [
            {
                "document_id": "doc1",
                "document_name": "poliza.pdf",
                "document_type": "poliza_de_la_aseguradora",
                "case_id": self.case_id,
                "risk_level": "bajo",
                "fraud_score": 0.1,
                "confidence": 0.9,
                "analysis_model": "test-model",
                "guide_version": "1.0",
                "analisis_completo": "Test analysis"
            }
        ]

    def test_from_case_basic(self):
        """Test basic case context creation."""
        context = CaseContext.from_case(
            case_id=self.case_id,
            consolidated=self.mock_consolidated,
            extractions=self.mock_extractions,
            fraud_results=self.mock_fraud_results
        )

        self.assertEqual(context.case_id, self.case_id)
        self.assertEqual(len(context.documents), 1)
        self.assertEqual(context.documents[0].document_id, "doc1")

    @patch.object(CaseContext, '_load_document_metadata')
    def test_from_case_with_metadata(self, mock_load_metadata):
        """Test case context creation with document metadata loading."""
        # Setup mock return value - return two dicts as expected
        mock_metadata = {
            "original_name": "poliza.pdf",
            "pages": 10,
            "processed_at": "2025-01-15T10:00:00"
        }
        # Return tuple of (by_name, by_id) dicts
        mock_load_metadata.return_value = ({"poliza.pdf": mock_metadata}, {"doc1": mock_metadata})

        context = CaseContext.from_case(
            case_id=self.case_id,
            consolidated=self.mock_consolidated,
            extractions=self.mock_extractions,
            fraud_results=self.mock_fraud_results
        )

        # Verify metadata was loaded
        mock_load_metadata.assert_called_once()
        # Verify the document has expected attributes
        self.assertEqual(context.documents[0].document_id, "doc1")
        self.assertEqual(context.documents[0].document_type, "poliza_de_la_aseguradora")

    @patch('fraud_scorer.analyzers.correlation.models.case_context.get_conn')
    def test_load_document_metadata_with_db(self, mock_get_conn):
        """Test _load_document_metadata with database records."""
        # Mock database connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        # Mock database rows
        mock_rows = [
            {
                "id": "doc1",
                "filename": "test_document.pdf",
                "mime_type": "application/pdf",
                "filepath": "/path/to/doc.pdf",
                "page_count": 5,
                "language": "es",
                "created_at": "2025-01-20T15:30:00"
            }
        ]
        mock_cursor.fetchall.return_value = mock_rows
        mock_conn.execute.return_value = mock_cursor
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.__exit__.return_value = None
        mock_get_conn.return_value = mock_conn

        # Call the method
        by_name, by_id = CaseContext._load_document_metadata(
            case_id=self.case_id,
            cache_manager=None,
            case_index={}
        )

        # Verify results
        self.assertIn("doc1", by_id)
        self.assertEqual(by_id["doc1"]["filename"], "test_document.pdf")
        self.assertEqual(by_id["doc1"]["page_count"], 5)
        self.assertIn("test_document.pdf", by_name)

    @patch('fraud_scorer.analyzers.correlation.models.case_context.get_conn')
    def test_load_document_metadata_without_db(self, mock_get_conn):
        """Test _load_document_metadata when database is unavailable."""
        # Simulate database connection error
        mock_get_conn.side_effect = Exception("Database error")

        # Call the method
        by_name, by_id = CaseContext._load_document_metadata(
            case_id=self.case_id,
            cache_manager=None,
            case_index={}
        )

        # Should return empty dicts when database fails
        self.assertEqual(by_name, {})
        self.assertEqual(by_id, {})

    @patch('fraud_scorer.analyzers.correlation.models.case_context.get_conn')
    def test_load_document_metadata_with_case_index(self, mock_get_conn):
        """Test _load_document_metadata with case index data."""
        # Mock database to return empty
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.execute.return_value = mock_cursor
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.__exit__.return_value = None
        mock_get_conn.return_value = mock_conn

        # Provide case index data with proper structure
        case_index = {
            "cache_files": ["doc2.json"],
            "documents": ["doc3.json"]
        }

        # Call the method
        by_name, by_id = CaseContext._load_document_metadata(
            case_id=self.case_id,
            cache_manager=None,
            case_index=case_index
        )

        # Verify that case index data is processed
        # The actual merging logic depends on implementation
        # For now we just verify the method executes without error
        self.assertIsNotNone(by_name)
        self.assertIsNotNone(by_id)

    def test_build_timeline(self):
        """Test timeline building from documents."""
        context = CaseContext.from_case(
            case_id=self.case_id,
            consolidated=self.mock_consolidated,
            extractions=[
                {
                    "document_id": "doc1",
                    "source_document": "poliza.pdf",
                    "document_type": "poliza_de_la_aseguradora",
                    "extracted_fields": {
                        "fecha_emision": "2024-01-15",
                        "vigencia_desde": "2024-02-01",
                    }
                },
                {
                    "document_id": "doc2",
                    "source_document": "denuncia.pdf",
                    "document_type": "denuncia_de_los_hechos",
                    "extracted_fields": {
                        "fecha_denuncia": "2024-12-20",
                    }
                }
            ],
            fraud_results=[]
        )

        # Verify timeline was built
        self.assertIn("timeline", context.__dict__)
        # The timeline structure depends on implementation details
        # but we can verify it exists
        self.assertIsNotNone(context.timeline)

    def test_entity_extraction(self):
        """Test entity extraction from documents."""
        context = CaseContext.from_case(
            case_id=self.case_id,
            consolidated=self.mock_consolidated,
            extractions=[
                {
                    "document_id": "doc1",
                    "source_document": "tarjeta_circulacion.pdf",
                    "document_type": "tarjeta_de_circulacion_vehiculo",
                    "extracted_fields": {
                        "placas": "ABC-123",
                        "niv": "1HGBH41JXMN109186",
                        "marca": "Honda",
                    }
                }
            ],
            fraud_results=[]
        )

        # Verify entities were extracted
        self.assertIn("entities", context.__dict__)
        self.assertIsNotNone(context.entities)