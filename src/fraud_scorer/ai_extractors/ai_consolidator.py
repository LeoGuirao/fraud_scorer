# src/fraud_scorer/ai_extractors/ai_consolidator.py

"""
AIConsolidator: Consolida extracciones múltiples usando razonamiento de IA
"""
import os  # ← Agregar este import
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import asyncio

from openai import AsyncOpenAI
import instructor
from pydantic import BaseModel, Field

from .config import ExtractionConfig, FieldPriority
from .models.extraction_models import (
    DocumentExtraction, 
    ConsolidatedExtraction,
    ExtractionBatch
)
from .prompts.consolidation_prompts import ConsolidationPromptBuilder

logger = logging.getLogger(__name__)

class ConsolidationDecision(BaseModel):
    """Modelo para las decisiones de consolidación"""
    field_name: str
    selected_value: Any
    source_document: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    alternatives_considered: List[Dict[str, Any]] = Field(default_factory=list)

class AIConsolidator:
    """
    Consolidador inteligente que usa IA para resolver conflictos
    y generar el conjunto final de datos
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Inicializa el consolidador
        """
        self.client = instructor.patch(
            AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        )
        self.config = ExtractionConfig()
        self.prompt_builder = ConsolidationPromptBuilder()
        
        # Cargar ejemplos de consolidación exitosa
        self.golden_examples = self._load_golden_examples()
        
        logger.info("AIConsolidator inicializado")
    
    async def consolidate_extractions(
        self,
        extractions: List[DocumentExtraction],
        case_id: str,
        use_advanced_reasoning: bool = True
    ) -> ConsolidatedExtraction:
        """
        Consolida múltiples extracciones en un resultado final
        
        Args:
            extractions: Lista de extracciones individuales
            case_id: ID del caso
            use_advanced_reasoning: Si usar el modelo avanzado de razonamiento
            
        Returns:
            ConsolidatedExtraction con los datos finales
        """
        logger.info(f"Consolidando {len(extractions)} extracciones para caso {case_id}")
        
        # Agrupar extracciones por campo
        field_groups = self._group_by_field(extractions)
        
        # Resolver cada campo
        consolidated_fields = {}
        consolidation_sources = {}
        conflicts_resolved = []
        confidence_scores = {}
        
        for field_name in self.config.REQUIRED_FIELDS:
            if field_name not in field_groups:
                consolidated_fields[field_name] = None
                confidence_scores[field_name] = 0.0
                continue
            
            # Obtener todas las opciones para este campo
            options = field_groups[field_name]
            
            if len(options) == 0:
                consolidated_fields[field_name] = None
                confidence_scores[field_name] = 0.0
            elif len(options) == 1:
                # Sin conflicto
                consolidated_fields[field_name] = options[0]['value']
                consolidation_sources[field_name] = f"Único valor de {options[0]['source']}"
                confidence_scores[field_name] = 1.0
            else:
                # Hay conflicto - usar IA para resolver
                if use_advanced_reasoning:
                    decision = await self._resolve_conflict_with_ai(
                        field_name=field_name,
                        options=options,
                        all_extractions=extractions
                    )
                else:
                    decision = self._resolve_conflict_with_rules(
                        field_name=field_name,
                        options=options
                    )
                
                consolidated_fields[field_name] = decision.selected_value
                consolidation_sources[field_name] = decision.reasoning
                confidence_scores[field_name] = decision.confidence
                
                if len(options) > 1:
                    conflicts_resolved.append({
                        "field": field_name,
                        "options": options,
                        "selected": decision.selected_value,
                        "reasoning": decision.reasoning
                    })
        
        # Validación final con IA
        if use_advanced_reasoning:
            consolidated_fields = await self._validate_with_ai(
                consolidated_fields, 
                extractions
            )
        
        # Crear resultado consolidado
        result = ConsolidatedExtraction(
            case_id=case_id,
            consolidated_fields=consolidated_fields,
            consolidation_sources=consolidation_sources,
            conflicts_resolved=conflicts_resolved,
            confidence_scores=confidence_scores
        )
        
        # Log de métricas
        self._log_consolidation_metrics(result)
        
        return result
    
    async def _resolve_conflict_with_ai(
        self,
        field_name: str,
        options: List[Dict[str, Any]],
        all_extractions: List[DocumentExtraction]
    ) -> ConsolidationDecision:
        """
        Usa IA para resolver conflictos entre valores
        """
        # Construir prompt para resolución
        prompt = self.prompt_builder.build_conflict_resolution_prompt(
            field_name=field_name,
            options=options,
            field_rules=self.config.FIELD_SOURCE_RULES.get(field_name, []),
            golden_examples=self._get_relevant_examples(field_name),
            context=self._build_context(all_extractions)
        )
        
        try:
            # Llamar a la IA para decisión
            response = await self.client.chat.completions.create(
                model=self.config.get_model_for_task("consolidation"),
                messages=[
                    {
                        "role": "system", 
                        "content": "Eres un experto ajustador de seguros con 20 años de experiencia."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_model=ConsolidationDecision,
                temperature=0.1,  # Muy baja para consistencia
                max_tokens=1000
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error en resolución con IA para {field_name}: {e}")
            # Fallback a reglas
            return self._resolve_conflict_with_rules(field_name, options)
    
    def _resolve_conflict_with_rules(
        self,
        field_name: str,
        options: List[Dict[str, Any]]
    ) -> ConsolidationDecision:
        """
        Resuelve conflictos usando reglas predefinidas (fallback)
        """
        # Obtener reglas de prioridad para este campo
        priority_sources = self.config.FIELD_SOURCE_RULES.get(field_name, [])
        
        # Si hay reglas, aplicarlas
        if priority_sources:
            for priority_source in priority_sources:
                for option in options:
                    if priority_source in option['source'].lower():
                        return ConsolidationDecision(
                            field_name=field_name,
                            selected_value=option['value'],
                            source_document=option['source'],
                            confidence=0.8,
                            reasoning=f"Seleccionado de {option['source']} por regla de prioridad",
                            alternatives_considered=options
                        )
        
        # Si no hay reglas o no aplican, tomar el más común
        from collections import Counter
        value_counts = Counter(opt['value'] for opt in options)
        most_common_value = value_counts.most_common(1)[0][0]
        
        source = next(opt['source'] for opt in options if opt['value'] == most_common_value)
        
        return ConsolidationDecision(
            field_name=field_name,
            selected_value=most_common_value,
            source_document=source,
            confidence=0.6,
            reasoning=f"Valor más común entre {len(options)} documentos",
            alternatives_considered=options
        )
    
    async def _validate_with_ai(
        self,
        consolidated_fields: Dict[str, Any],
        original_extractions: List[DocumentExtraction]
    ) -> Dict[str, Any]:
        """
        Validación final y ajustes con IA
        """
        prompt = self.prompt_builder.build_validation_prompt(
            consolidated_fields=consolidated_fields,
            original_extractions=[e.dict() for e in original_extractions]
        )
        
        try:
            # Pedir a la IA que valide y ajuste si es necesario
            response = await self.client.chat.completions.create(
                model=self.config.get_model_for_task("consolidation"),
                messages=[
                    {
                        "role": "system",
                        "content": "Valida que los datos consolidados sean coherentes y correctos."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            # Parsear respuesta
            content = response.choices[0].message.content
            if "VALIDADO_OK" in content:
                return consolidated_fields
            
            # Si hay ajustes sugeridos, aplicarlos
            if "AJUSTES:" in content:
                adjustments = self._parse_adjustments(content)
                for field, new_value in adjustments.items():
                    logger.info(f"Ajustando {field}: {consolidated_fields.get(field)} -> {new_value}")
                    consolidated_fields[field] = new_value
            
            return consolidated_fields
            
        except Exception as e:
            logger.error(f"Error en validación con IA: {e}")
            return consolidated_fields
    
    def _group_by_field(self, extractions: List[DocumentExtraction]) -> Dict[str, List[Dict]]:
        """
        Agrupa los valores extraídos por campo
        """
        field_groups = {}
        
        for extraction in extractions:
            for field_name, value in extraction.extracted_fields.items():
                if value is not None:  # Ignorar valores nulos
                    if field_name not in field_groups:
                        field_groups[field_name] = []
                    
                    field_groups[field_name].append({
                        'value': value,
                        'source': extraction.source_document,
                        'document_type': extraction.document_type
                    })
        
        return field_groups
    
    def _build_context(self, extractions: List[DocumentExtraction]) -> str:
        """
        Construye contexto adicional para la IA
        """
        context_parts = []
        
        # Resumen de documentos procesados
        doc_types = [e.document_type for e in extractions]
        context_parts.append(f"Documentos analizados: {', '.join(set(doc_types))}")
        
        # Contar campos encontrados
        field_counts = {}
        for e in extractions:
            for field, value in e.extracted_fields.items():
                if value is not None:
                    field_counts[field] = field_counts.get(field, 0) + 1
        
        context_parts.append(f"Campos con datos: {field_counts}")
        
        return "\n".join(context_parts)
    
    def _get_relevant_examples(self, field_name: str) -> List[Dict]:
        """
        Obtiene ejemplos relevantes para un campo específico
        """
        if field_name in self.golden_examples:
            return self.golden_examples[field_name][:3]  # Máximo 3 ejemplos
        return []
    
    def _load_golden_examples(self) -> Dict[str, List[Dict]]:
        """
        Carga ejemplos de consolidaciones exitosas anteriores
        """
        # Aquí cargarías desde data/training_examples/consolidation_examples.json
        return {
            "numero_poliza": [
                {
                    "options": [
                        {"value": "AX-123-B", "source": "poliza.pdf"},
                        {"value": "AX123B", "source": "factura.pdf"}
                    ],
                    "correct_choice": "AX-123-B",
                    "reasoning": "Formato de póliza.pdf es el estándar"
                }
            ]
        }
    
    def _parse_adjustments(self, ai_response: str) -> Dict[str, Any]:
        """
        Parsea ajustes sugeridos por la IA
        """
        adjustments = {}
        
        # Buscar patrón de ajustes en la respuesta
        lines = ai_response.split('\n')
        in_adjustments = False
        
        for line in lines:
            if "AJUSTES:" in line:
                in_adjustments = True
                continue
            
            if in_adjustments and ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    field = parts[0].strip()
                    value = parts[1].strip()
                    adjustments[field] = value
        
        return adjustments
    
    def _log_consolidation_metrics(self, result: ConsolidatedExtraction):
        """
        Registra métricas de la consolidación
        """
        fields_filled = sum(
            1 for v in result.consolidated_fields.values() 
            if v is not None
        )
        total_fields = len(result.consolidated_fields)
        conflicts = len(result.conflicts_resolved)
        avg_confidence = sum(result.confidence_scores.values()) / len(result.confidence_scores) if result.confidence_scores else 0
        
        logger.info(
            f"Consolidación completada | "
            f"Caso: {result.case_id} | "
            f"Campos: {fields_filled}/{total_fields} | "
            f"Conflictos resueltos: {conflicts} | "
            f"Confianza promedio: {avg_confidence:.2%}"
        )