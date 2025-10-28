#!/usr/bin/env python3
"""
Script mejorado para validar la capacidad del Agente Rick en modo agéntico
para correlacionar información ESPECÍFICA entre la denuncia de hechos y datos GPS.
"""

import json
import logging
from pathlib import Path
from fraud_scorer.ai.orchestration import AgenteRickService
from fraud_scorer.ai.config import load_config

# Configurar logging más detallado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_agent_validation():
    """Prueba detallada de validación cruzada entre denuncia y GPS."""

    # Cargar configuración con modo agéntico activado
    config_overrides = {
        "agent_mode_enabled": True,
        "agent_max_iterations": 10,  # Más iteraciones para análisis complejo
        "agent_verbose": True,
        "agent_gps_preview_limit": 100,  # Más registros GPS
        "max_results": 20  # Más documentos para búsqueda
    }

    config = load_config(overrides=config_overrides)
    logger.info("✅ Modo agéntico activado con configuración extendida")

    # Inicializar servicio
    service = AgenteRickService(config=config)
    logger.info("Servicio AgenteRick inicializado")

    # Caso de prueba
    case_id = "CASE-2025-0001"

    # Preguntas más específicas que fuerzan al agente a:
    # 1. Buscar en la denuncia de hechos
    # 2. Extraer ubicación y hora exacta
    # 3. Validar contra GPS
    # 4. Dar conclusión detallada
    questions = [
        """Busca específicamente en el documento de DENUNCIA DE HECHOS del caso y extrae:
        1. La ubicación EXACTA donde se declaró que ocurrió el siniestro (carretera, kilómetro, municipio, estado)
        2. La fecha y hora EXACTA declarada del siniestro
        3. Los vehículos involucrados (placas, números económicos)

        Después, consulta los datos GPS del documento '16 1 Monitoreo 18AT9H.pdf' para esa fecha específica y valida si coincide.""",

        """Analiza la DENUNCIA DE HECHOS donde se describe el lugar del siniestro.
        Luego revisa el monitoreo GPS '16 1 Monitoreo 18AT9H.pdf' del 13 de febrero de 2024.

        Compara:
        1. ¿El vehículo estuvo en el lugar declarado (carretera y kilómetro específico)?
        2. ¿A qué hora estuvo ahí según el GPS?
        3. ¿Las coordenadas GPS coinciden con la ubicación declarada?

        Da una conclusión CLARA: ¿COINCIDE o NO COINCIDE? y explica por qué con evidencia específica.""",

        """Según la DENUNCIA DE HECHOS, identifica la hora exacta del siniestro declarado.
        Luego consulta el GPS '16 1 Monitoreo 18AT9H.pdf' y verifica:

        1. Dónde estaba el vehículo a esa hora exacta (coordenadas y ubicación)
        2. Si hubo paradas o eventos anormales

        Concluye si la información GPS VALIDA o CONTRADICE lo declarado en la denuncia."""
    ]

    logger.info("\n" + "="*80)
    logger.info("INICIANDO PRUEBAS DE VALIDACIÓN CRUZADA")
    logger.info("="*80)

    for i, question in enumerate(questions, 1):
        logger.info("\n" + "-"*60)
        logger.info(f"PREGUNTA {i}: {question}")
        logger.info("-"*60)

        try:
            # Ejecutar consulta
            result = service.query(
                case_id=case_id,
                question=question,
                user_id="test_user",
                scope="case",
                language="es",
                module="validation_test"
            )

            # Mostrar resultados
            # Mostrar respuesta completa
            logger.info("\n" + "-"*60)
            logger.info("📝 RESPUESTA DEL AGENTE:")
            logger.info("-"*60)
            logger.info(result.answer)

            # Buscar conclusión de coincidencia en la respuesta
            answer_lower = result.answer.lower()
            if "no coincide" in answer_lower or "contradice" in answer_lower:
                logger.info("\n⚠️ RESULTADO: NO COINCIDE - Posible inconsistencia detectada")
            elif "coincide" in answer_lower or "valida" in answer_lower:
                logger.info("\n✅ RESULTADO: COINCIDE - Información validada")
            else:
                logger.info("\n❓ RESULTADO: No concluyente")

            # Detalles de metadata
            if result.metadata:
                logger.info("\n📊 DETALLES TÉCNICOS:")
                logger.info("- Modo: %s", result.metadata.get("mode"))
                logger.info("- Documentos analizados: %s", result.metadata.get("retrieved", 0))

                # Mostrar pasos del agente con más detalle
                if result.metadata.get("agent_steps"):
                    logger.info("\n🔧 PASOS DEL AGENTE:")
                    for idx, step in enumerate(result.metadata["agent_steps"], 1):
                        logger.info(f"\n  Paso {idx}:")
                        logger.info(f"    Herramienta: {step.get('tool')}")
                        if step.get("tool_input"):
                            input_str = step["tool_input"]
                            if len(input_str) > 200:
                                input_str = input_str[:200] + "..."
                            logger.info(f"    Input: {input_str}")

                # Mostrar consultas GPS con filtros
                if result.metadata.get("gps_queries"):
                    logger.info("\n🛰️ CONSULTAS GPS REALIZADAS:")
                    for idx, gps_query in enumerate(result.metadata["gps_queries"], 1):
                        logger.info(f"\n  Consulta GPS {idx}:")
                        logger.info(f"    Documento: {gps_query.get('document_name')}")
                        logger.info(f"    Registros: {gps_query.get('row_count', 0)}")
                        if gps_query.get("filters"):
                            filters = gps_query["filters"]
                            if filters.get("start_time"):
                                logger.info(f"    Desde: {filters['start_time']}")
                            if filters.get("end_time"):
                                logger.info(f"    Hasta: {filters['end_time']}")

            # Fuentes utilizadas con deduplicación
            if result.sources:
                logger.info("\n📚 FUENTES CONSULTADAS:")
                unique_sources = {}
                for source in result.sources:
                    doc_name = source.get("source_document", "Unknown")
                    doc_type = source.get("document_type", "Unknown")
                    key = f"{doc_name}|{doc_type}"
                    if key not in unique_sources:
                        unique_sources[key] = source

                for source in unique_sources.values():
                    logger.info(f"  - {source.get('source_document')} (tipo: {source.get('document_type')})")

            logger.info("\n⏱️ Tiempo de respuesta: %d ms", result.latency_ms)

        except Exception as e:
            logger.error("Error ejecutando consulta: %s", e, exc_info=True)

    # Revisar el log de auditoría
    audit_path = config.audit_log_path
    if audit_path.exists():
        logger.info("\n" + "="*80)
        logger.info("REVISANDO AUDITORÍA")
        logger.info("="*80)

        with open(audit_path, 'r') as f:
            lines = f.readlines()
            # Mostrar últimas entradas con agent_steps
            for line in lines[-5:]:
                try:
                    entry = json.loads(line)
                    if entry.get("agent_steps"):
                        logger.info("\nEntrada de auditoría con pasos de agente:")
                        logger.info("- Timestamp: %s", entry.get("timestamp"))
                        logger.info("- Status: %s", entry.get("status"))
                        logger.info("- Agent steps: %d", len(entry.get("agent_steps", [])))
                except:
                    pass

if __name__ == "__main__":
    test_agent_validation()