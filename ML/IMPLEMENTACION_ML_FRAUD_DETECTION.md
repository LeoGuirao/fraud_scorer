# 🤖 IMPLEMENTACIÓN MACHINE LEARNING PARA DETECCIÓN DE FRAUDE
## Basado en Análisis de Reportes Reales y Verificaciones Manuales

**Versión:** 1.0
**Fecha:** Septiembre 2025
**Autor:** Leonardo Guirao - CTO
**Basado en:** Análisis de reportes reales de HDI Seguros + Sistema de verificaciones manuales

---

## 📋 RESUMEN EJECUTIVO

Este documento define la estrategia completa de Machine Learning para el Sistema de Análisis de Siniestros, basada en el análisis exhaustivo de reportes reales de HDI Seguros y diseñada para funcionar con el sistema de verificaciones manuales implementado.

### 🎯 Hallazgos Clave de Reportes Reales Analizados

**Casos Analizados:**
- ✅ **CON TENTATIVA**: Caso Castro (proyector), Caso MODA YKT (robo), Caso Aceros Ocotlán
- ✅ **SIN TENTATIVA**: Caso Chihuahua Meat (robo legítimo)

**Patrones de Fraude Identificados:**
1. **Documentos apócrifos** (100% de detección de fraude)
2. **Inconsistencias técnicas** (diagnósticos imposibles)
3. **Discrepancias vehiculares** (placas, NIV, REPUVE)
4. **Temporalidad sospechosa** (reportes tempranos)
5. **Validaciones fallidas** (SAT, REPUVE, SCT, Fiscalías)

---

## 🔍 ANÁLISIS DE PATRONES EN REPORTES REALES

### Caso 1: Felipe Castro - DAÑO POR VOLTAJE (CON TENTATIVA)

#### **Indicadores de Fraude Detectados:**
```json
{
  "caso_id": "20250000004129",
  "verdict": "CON_TENTATIVA",
  "fraud_score": 0.95,
  "indicadores_criticos": [
    {
      "tipo": "error_tecnico_grave",
      "descripcion": "Diagnóstico menciona 'Lámpara' en proyector láser",
      "peso": 1.0,
      "evidencia": "Epson confirmó que modelo EH-LS300B no tiene lámpara"
    },
    {
      "tipo": "evidencia_fisica_ausente",
      "descripcion": "Sin olor a quemado tras supuesto 'tronido'",
      "peso": 0.8,
      "explicacion": "Arco eléctrico siempre deja olor a quemado"
    },
    {
      "tipo": "reporte_temprano",
      "descripcion": "14 días desde inicio de póliza",
      "peso": 0.4
    },
    {
      "tipo": "mantenimiento_deficiente",
      "descripcion": "Exceso polvo + configuración incorrecta altitud",
      "peso": 0.6
    }
  ]
}
```

#### **Verificaciones Manuales Realizadas:**
- ✅ **Epson Service Center**: Confirmó que modelo no tiene lámpara
- ✅ **Manual del usuario**: Requiere modo alta altitud en CDMX
- ✅ **Análisis técnico**: Falta mantenimiento evidente

### Caso 2: MODA YKT - ROBO DE MERCANCÍA (CON TENTATIVA)

#### **Indicadores de Fraude Detectados:**
```json
{
  "caso_id": "20250000002494",
  "verdict": "CON_TENTATIVA",
  "fraud_score": 0.98,
  "indicadores_criticos": [
    {
      "tipo": "documento_apocrifo_confirmado",
      "descripcion": "Cartas falsas de Transportes Medina",
      "peso": 1.0,
      "evidencia": "Transportista desconoció documentos por teléfono"
    },
    {
      "tipo": "placas_inconsistentes",
      "descripcion": "Carta Porte vs otros documentos",
      "peso": 0.95,
      "detalle": {
        "carta_porte": ["734EH6", "74UW1B"],
        "otros_docs": ["07BE4W", "63UW1A"]
      }
    },
    {
      "tipo": "discrepancia_repuve",
      "descripcion": "Marca KENWORTH vs PETERBILT",
      "peso": 0.7,
      "niv": "1XP7DB9X86D893212"
    }
  ]
}
```

#### **Verificaciones Manuales Realizadas:**
- ✅ **Llamada a transportista**: Confirmó falsificación de documentos
- ✅ **REPUVE**: Discrepancia de marca vehicular
- ✅ **Análisis GPS**: Ruta coherente pero documentos falsos

### Caso 3: Aceros Ocotlán - ROBO DE VARILLAS (CON TENTATIVA)

#### **Indicadores de Fraude Detectados:**
```json
{
  "caso_id": "20240000001361",
  "verdict": "CON_TENTATIVA",
  "fraud_score": 0.92,
  "indicadores_criticos": [
    {
      "tipo": "unidad_robo_previo",
      "descripcion": "Vehículo robado 19 días antes del siniestro",
      "peso": 0.95,
      "fechas": {
        "robo_previo": "25/01/2024",
        "siniestro_actual": "13/02/2024"
      }
    },
    {
      "tipo": "licencias_vencidas",
      "descripcion": "Operadores sin facultades legales",
      "peso": 0.5,
      "detalle": "Licencia vencida + falta examen médico"
    }
  ]
}
```

### Caso 4: Chihuahua Meat - ROBO DE CARNE (SIN TENTATIVA)

#### **Características de Caso Legítimo:**
```json
{
  "caso_id": "20250000003818",
  "verdict": "SIN_TENTATIVA",
  "fraud_score": 0.15,
  "validaciones_exitosas": [
    {
      "tipo": "documentacion_consistente",
      "descripcion": "Todos los documentos coherentes entre sí"
    },
    {
      "tipo": "validaciones_gubernamentales_ok",
      "detalle": {
        "sat_cfdi": "vigente",
        "repuve_status": "con_reporte_robo_post_siniestro",
        "fiscalia_validada": true
      }
    },
    {
      "tipo": "ruta_coherente",
      "descripcion": "Trayectoria lógica y sin desvíos"
    }
  ]
}
```

---

## 🎯 ESTRATEGIA ML BASADA EN HALLAZGOS REALES

### 1. **Clasificador Multi-Nivel por Tipo de Siniestro**

```python
class FraudClassifierV2:
    """
    Clasificador especializado basado en patrones reales identificados
    """
    def __init__(self):
        self.classifiers = {
            'voltaje': VoltageClaimClassifier(),
            'robo_transito': TransitTheftClassifier(),
            'robo_general': GeneralTheftClassifier()
        }

        # Pesos específicos por tipo basados en casos reales
        self.fraud_weights = {
            'voltaje': {
                'diagnostico_tecnico_imposible': 1.0,
                'sin_evidencia_fisica': 0.8,
                'mantenimiento_deficiente': 0.6,
                'reporte_temprano': 0.4
            },
            'robo_transito': {
                'documento_apocrifo': 1.0,
                'placas_inconsistentes': 0.95,
                'unidad_robo_previo': 0.95,
                'discrepancia_repuve': 0.7,
                'licencia_inadecuada': 0.5
            }
        }

    def classify(self, case_data, verification_results):
        case_type = case_data['tipo_siniestro']
        classifier = self.classifiers.get(case_type)

        if not classifier:
            return self.generic_classification(case_data, verification_results)

        # Extraer features específicas del tipo
        features = self.extract_features(case_data, verification_results, case_type)

        # Aplicar pesos específicos
        weighted_score = self.apply_weights(features, case_type)

        # Clasificar con umbral dinámico
        threshold = self.get_threshold_for_type(case_type)

        return {
            'fraud_probability': weighted_score,
            'classification': 'CON_TENTATIVA' if weighted_score > threshold else 'SIN_TENTATIVA',
            'confidence': self.calculate_confidence(features),
            'key_indicators': self.get_key_indicators(features, case_type)
        }

class VoltageClaimClassifier:
    """Especializado en daños por voltaje"""

    def extract_features(self, case_data, verifications):
        features = {}

        # Feature 1: Diagnóstico técnico coherente
        if 'diagnostico' in case_data:
            features['diagnostico_coherente'] = self.validate_technical_diagnosis(
                case_data['diagnostico'],
                case_data['modelo_equipo']
            )

        # Feature 2: Evidencia física presente
        if 'informe_ajustador' in case_data:
            features['evidencia_fisica'] = self.detect_physical_evidence(
                case_data['informe_ajustador']
            )

        # Feature 3: Mantenimiento adecuado
        features['mantenimiento_ok'] = self.assess_maintenance(case_data)

        # Feature 4: Días desde inicio póliza
        features['dias_desde_poliza'] = case_data.get('dias_desde_inicio_poliza', 999)

        return features

    def validate_technical_diagnosis(self, diagnosis, equipment_model):
        """Validar coherencia técnica del diagnóstico"""
        # Ejemplo: detectar componentes imposibles
        impossible_components = {
            'EH-LS300B': ['lampara', 'bulbo'],  # Proyector láser
            'LED_TV': ['tubo_rayos_catodicos'],
            'LAPTOP': ['diskette', 'cd_rom']
        }

        if equipment_model in impossible_components:
            for impossible in impossible_components[equipment_model]:
                if impossible.lower() in diagnosis.lower():
                    return 0.0  # Diagnóstico imposible

        return 1.0  # Diagnóstico posible

class TransitTheftClassifier:
    """Especializado en robos en tránsito"""

    def extract_features(self, case_data, verifications):
        features = {}

        # Feature 1: Consistencia de placas
        features['placas_consistentes'] = self.check_plate_consistency(case_data)

        # Feature 2: Estado REPUVE
        if 'repuve_result' in verifications:
            features['repuve_status'] = self.analyze_repuve_status(
                verifications['repuve_result'],
                case_data['fecha_siniestro']
            )

        # Feature 3: Documentos verificados
        features['documentos_validos'] = self.validate_documents(verifications)

        # Feature 4: Coherencia de ruta GPS
        if 'gps_data' in case_data:
            features['ruta_coherente'] = self.analyze_route_coherence(
                case_data['gps_data']
            )

        return features
```

### 2. **Sistema de Aprendizaje con Verificaciones Manuales**

```python
class ManualVerificationLearner:
    """
    Aprende de las verificaciones manuales para mejorar el sistema
    """

    def __init__(self):
        self.verification_patterns = {}
        self.fraud_correlation_matrix = {}
        self.validation_effectiveness = {}

    def learn_from_manual_verifications(self, cases_with_verifications):
        """
        Aprende patrones de las verificaciones manuales realizadas
        """
        for case in cases_with_verifications:
            # Aprender qué verificaciones son más efectivas
            self.update_validation_effectiveness(case)

            # Identificar correlaciones entre verificaciones y fraude
            self.update_fraud_correlations(case)

            # Mejorar templates de verificación
            self.improve_verification_templates(case)

    def update_validation_effectiveness(self, case):
        """Actualizar efectividad de cada tipo de verificación"""
        verdict = case['verdict']

        for verification in case['verifications']:
            v_type = verification['type']
            result = verification['result']

            if v_type not in self.validation_effectiveness:
                self.validation_effectiveness[v_type] = {
                    'total': 0,
                    'fraud_detected': 0,
                    'false_positives': 0,
                    'effectiveness_score': 0.0
                }

            self.validation_effectiveness[v_type]['total'] += 1

            # Si la verificación falló y hay fraude, es efectiva
            if result['status'] == 'failed' and verdict == 'CON_TENTATIVA':
                self.validation_effectiveness[v_type]['fraud_detected'] += 1

            # Si la verificación falló pero no hay fraude, es falso positivo
            elif result['status'] == 'failed' and verdict == 'SIN_TENTATIVA':
                self.validation_effectiveness[v_type]['false_positives'] += 1

            # Recalcular score de efectividad
            stats = self.validation_effectiveness[v_type]
            if stats['total'] > 0:
                precision = stats['fraud_detected'] / (stats['fraud_detected'] + stats['false_positives'] + 1)
                recall = stats['fraud_detected'] / max(stats['total'], 1)
                stats['effectiveness_score'] = 2 * (precision * recall) / (precision + recall + 0.001)

    def generate_smart_checklist(self, case_data):
        """
        Generar checklist inteligente basado en aprendizajes
        """
        checklist = []
        case_type = case_data['tipo_siniestro']

        # Obtener verificaciones más efectivas para este tipo
        effective_verifications = self.get_most_effective_verifications(case_type)

        for verification in effective_verifications:
            if self.should_include_verification(verification, case_data):
                checklist.append(self.create_verification_item(verification, case_data))

        return sorted(checklist, key=lambda x: x['priority'], reverse=True)

    def get_most_effective_verifications(self, case_type):
        """Obtener verificaciones más efectivas para un tipo de caso"""

        # Basado en casos reales analizados
        effective_by_type = {
            'robo_transito': [
                {'type': 'REPUVE', 'effectiveness': 0.95, 'examples': 'Caso Aceros - robo previo'},
                {'type': 'SAT', 'effectiveness': 0.90, 'examples': 'Caso MODA YKT - CFDI falso'},
                {'type': 'PHONE_VERIFICATION', 'effectiveness': 1.0, 'examples': 'Caso MODA YKT - transportista'},
                {'type': 'SCT', 'effectiveness': 0.70, 'examples': 'Licencias vencidas'}
            ],
            'voltaje': [
                {'type': 'MANUFACTURER', 'effectiveness': 1.0, 'examples': 'Caso Castro - Epson'},
                {'type': 'TECHNICAL_MANUAL', 'effectiveness': 0.8, 'examples': 'Configuración altitud'},
                {'type': 'PHYSICAL_INSPECTION', 'effectiveness': 0.9, 'examples': 'Olor a quemado'}
            ]
        }

        return effective_by_type.get(case_type, [])
```

### 3. **Pipeline de Entrenamiento con 60 Reportes**

```python
class TrainingPipelineV2:
    """
    Pipeline completo para entrenar con los 60 reportes reales + verificaciones manuales
    """

    def __init__(self):
        self.report_processor = ReportProcessor()
        self.feature_extractor = FeatureExtractor()
        self.verification_learner = ManualVerificationLearner()
        self.model_trainer = ModelTrainer()

    async def train_with_60_reports(self, reports_path, verification_logs):
        """
        Entrenar sistema completo con 60 reportes + logs de verificaciones
        """
        print("🔍 FASE 1: Procesando 60 reportes reales...")

        # 1. Procesar reportes de texto a estructura JSON
        structured_reports = await self.process_reports(reports_path)
        print(f"✅ Procesados {len(structured_reports)} reportes")

        # 2. Extraer patterns de fraude
        fraud_patterns = self.extract_fraud_patterns(structured_reports)
        print(f"✅ Identificados {len(fraud_patterns)} patrones de fraude")

        # 3. Aprender de verificaciones manuales
        verification_insights = self.verification_learner.learn_from_manual_verifications(
            verification_logs
        )
        print(f"✅ Analizadas {len(verification_logs)} sesiones de verificación")

        # 4. Crear datasets de entrenamiento
        training_data = self.create_training_datasets(
            structured_reports,
            fraud_patterns,
            verification_insights
        )

        # 5. Entrenar modelos especializados
        models = await self.train_specialized_models(training_data)

        # 6. Validación cruzada con casos reales
        validation_results = await self.validate_with_real_cases(models, structured_reports)

        # 7. Generar sistema de reglas heurísticas
        heuristic_rules = self.generate_heuristic_rules(fraud_patterns)

        return {
            'models': models,
            'validation_results': validation_results,
            'heuristic_rules': heuristic_rules,
            'verification_insights': verification_insights,
            'performance_metrics': self.calculate_performance_metrics(validation_results)
        }

    def extract_fraud_patterns(self, reports):
        """Extraer patrones específicos de los reportes reales"""
        patterns = {
            'determinantes': [],  # 100% indicativo de fraude
            'altamente_probable': [],  # 80-99%
            'sospechoso': [],  # 50-79%
            'correlacionado': []  # Requiere múltiples indicadores
        }

        for report in reports:
            if report['verdict'] == 'CON_TENTATIVA':
                for indicator in report['fraud_indicators']:
                    if indicator['confidence'] >= 0.95:
                        patterns['determinantes'].append(indicator)
                    elif indicator['confidence'] >= 0.8:
                        patterns['altamente_probable'].append(indicator)
                    elif indicator['confidence'] >= 0.5:
                        patterns['sospechoso'].append(indicator)

        # Consolidar patrones similares
        return self.consolidate_patterns(patterns)

    def create_training_datasets(self, reports, patterns, verification_insights):
        """Crear datasets optimizados para entrenamiento"""

        # Dataset 1: Clasificación binaria CON/SIN TENTATIVA
        binary_dataset = []

        # Dataset 2: Scoring de fraude (0-1)
        scoring_dataset = []

        # Dataset 3: Detección de verificaciones necesarias
        verification_dataset = []

        for report in reports:
            # Features básicas
            features = self.feature_extractor.extract_basic_features(report)

            # Features de verificaciones
            verification_features = self.feature_extractor.extract_verification_features(
                report, verification_insights
            )

            # Combinar features
            combined_features = {**features, **verification_features}

            # Binary classification
            binary_dataset.append({
                'features': combined_features,
                'label': 1 if report['verdict'] == 'CON_TENTATIVA' else 0
            })

            # Fraud scoring
            scoring_dataset.append({
                'features': combined_features,
                'score': report['fraud_score']
            })

            # Verification prediction
            verification_dataset.append({
                'document_types': report['document_types'],
                'case_type': report['case_type'],
                'needed_verifications': report['required_verifications']
            })

        return {
            'binary': binary_dataset,
            'scoring': scoring_dataset,
            'verification': verification_dataset
        }
```

### 4. **Sistema de Features Basado en Casos Reales**

```python
class FeatureExtractorV2:
    """
    Extractor de features basado en análisis de casos reales
    """

    def extract_comprehensive_features(self, case_data, verification_results):
        """Extraer features completas basadas en casos reales"""

        features = {}

        # 1. Features temporales (importantes en casos reales)
        features.update(self.extract_temporal_features(case_data))

        # 2. Features de consistencia documental (críticas)
        features.update(self.extract_consistency_features(case_data))

        # 3. Features de verificaciones externas
        features.update(self.extract_verification_features(verification_results))

        # 4. Features técnicas específicas por tipo
        features.update(self.extract_technical_features(case_data))

        # 5. Features de correlación entre documentos
        features.update(self.extract_correlation_features(case_data))

        return features

    def extract_temporal_features(self, case_data):
        """Features temporales críticas identificadas"""
        features = {}

        # Días desde inicio de póliza (importante en caso Castro)
        features['dias_desde_poliza'] = case_data.get('dias_desde_inicio_poliza', 999)
        features['reporte_muy_temprano'] = 1 if features['dias_desde_poliza'] < 30 else 0
        features['reporte_sospechoso'] = 1 if features['dias_desde_poliza'] < 15 else 0

        # Demora en reporte
        features['dias_demora_reporte'] = case_data.get('dias_entre_siniestro_reporte', 0)
        features['demora_excesiva'] = 1 if features['dias_demora_reporte'] > 45 else 0

        # Fechas de documentos vs siniestro
        if 'fechas_documentos' in case_data:
            fechas = case_data['fechas_documentos']
            fecha_siniestro = case_data['fecha_siniestro']

            features['docs_posteriores_siniestro'] = sum(
                1 for fecha in fechas if fecha > fecha_siniestro
            )

        return features

    def extract_consistency_features(self, case_data):
        """Features de consistencia entre documentos"""
        features = {}

        # Consistencia de placas (crítico en caso MODA YKT)
        if 'placas_documentos' in case_data:
            placas_unicas = set(case_data['placas_documentos'])
            features['placas_inconsistentes'] = 1 if len(placas_unicas) > 1 else 0
            features['numero_placas_diferentes'] = len(placas_unicas)

        # Consistencia de NIV/VIN
        if 'niv_documentos' in case_data:
            niv_unico = len(set(case_data['niv_documentos'])) == 1
            features['niv_consistente'] = 1 if niv_unico else 0

        # Consistencia de importes
        if 'importes_documentos' in case_data:
            importes = case_data['importes_documentos']
            if len(importes) > 1:
                max_diff = max(importes) - min(importes)
                features['diferencia_importes'] = max_diff
                features['importes_muy_diferentes'] = 1 if max_diff > 100000 else 0

        return features

    def extract_verification_features(self, verification_results):
        """Features basadas en verificaciones manuales"""
        features = {}

        if not verification_results:
            features['verificaciones_realizadas'] = 0
            return features

        total_verifications = len(verification_results)
        failed_verifications = sum(1 for v in verification_results if v['status'] == 'failed')

        features['verificaciones_realizadas'] = total_verifications
        features['verificaciones_fallidas'] = failed_verifications
        features['porcentaje_fallas'] = failed_verifications / total_verifications if total_verifications > 0 else 0

        # Features específicas por tipo de verificación
        for verification in verification_results:
            v_type = verification['type']
            status = verification['status']

            features[f'{v_type.lower()}_failed'] = 1 if status == 'failed' else 0

            # Features específicas basadas en casos reales
            if v_type == 'SAT' and status == 'failed':
                features['cfdi_invalido'] = 1
                if 'no_existe' in verification.get('detail', ''):
                    features['cfdi_no_existe'] = 1

            elif v_type == 'REPUVE' and status == 'failed':
                features['repuve_problema'] = 1
                if 'robo_previo' in verification.get('detail', ''):
                    features['unidad_robo_previo'] = 1

            elif v_type == 'PHONE_VERIFICATION' and status == 'failed':
                features['documento_apocrifo_confirmado'] = 1  # Muy importante

        return features

    def extract_technical_features(self, case_data):
        """Features técnicas específicas por tipo de siniestro"""
        features = {}

        case_type = case_data.get('tipo_siniestro', '').lower()

        if 'voltaje' in case_type or 'electrico' in case_type:
            features.update(self.extract_electrical_features(case_data))
        elif 'robo' in case_type:
            features.update(self.extract_theft_features(case_data))

        return features

    def extract_electrical_features(self, case_data):
        """Features específicas para daños eléctricos (basado en caso Castro)"""
        features = {}

        # Diagnóstico técnico coherente
        if 'diagnostico' in case_data:
            diagnostico = case_data['diagnostico'].lower()

            # Componentes imposibles detectados
            impossible_components = ['lampara', 'bulbo', 'tubo_rayos_catodicos']
            features['componente_imposible'] = any(comp in diagnostico for comp in impossible_components)

            # Términos técnicos esperados
            expected_terms = ['quemado', 'sobrecalentamiento', 'corto', 'circuito']
            features['terminos_tecnicos_presentes'] = sum(1 for term in expected_terms if term in diagnostico)

        # Evidencia física
        if 'informe_ajustador' in case_data:
            informe = case_data['informe_ajustador'].lower()
            features['olor_quemado'] = 1 if 'olor' in informe and 'quemado' in informe else 0
            features['marcas_quemadura'] = 1 if 'marca' in informe and 'quema' in informe else 0
            features['polvo_excesivo'] = 1 if 'polvo' in informe or 'suciedad' in informe else 0

        return features
```

### 5. **Generador de Reglas Heurísticas Basadas en Casos Reales**

```python
class HeuristicRuleGenerator:
    """
    Genera reglas heurísticas basadas en los patrones identificados en casos reales
    """

    def generate_rules_from_real_cases(self):
        """Generar reglas basadas en análisis de casos reales"""

        rules = {
            'determinant_rules': [],  # Reglas que determinan fraude automáticamente
            'high_probability_rules': [],  # Reglas de alta probabilidad
            'correlation_rules': []  # Reglas que requieren múltiples condiciones
        }

        # REGLAS DETERMINANTES (basadas en casos reales)

        # Regla 1: Documento apócrifo confirmado (Caso MODA YKT)
        rules['determinant_rules'].append({
            'id': 'RULE_001',
            'name': 'documento_apocrifo_confirmado',
            'description': 'Documento confirmado como falso por tercero',
            'condition': 'phone_verification_failed AND documento_desconocido_por_emisor',
            'action': 'SET fraud_score = 1.0',
            'evidence_from': 'Caso MODA YKT - Transportes Medina',
            'confidence': 1.0
        })

        # Regla 2: Diagnóstico técnico imposible (Caso Castro)
        rules['determinant_rules'].append({
            'id': 'RULE_002',
            'name': 'diagnostico_tecnico_imposible',
            'description': 'Diagnóstico menciona componente inexistente',
            'condition': 'componente_imposible AND tipo_siniestro = voltaje',
            'action': 'SET fraud_score = 0.95',
            'evidence_from': 'Caso Castro - Lámpara en proyector láser',
            'confidence': 0.95
        })

        # REGLAS DE ALTA PROBABILIDAD

        # Regla 3: Unidad con robo previo (Caso Aceros)
        rules['high_probability_rules'].append({
            'id': 'RULE_003',
            'name': 'unidad_robo_previo_reciente',
            'description': 'Vehículo reportado robado antes del siniestro',
            'condition': 'repuve_robo_previo AND dias_diferencia < 60',
            'action': 'ADD fraud_score += 0.9',
            'evidence_from': 'Caso Aceros - Robo 19 días antes',
            'confidence': 0.9
        })

        # Regla 4: Placas inconsistentes (Caso MODA YKT)
        rules['high_probability_rules'].append({
            'id': 'RULE_004',
            'name': 'placas_multiples_documentos',
            'description': 'Diferentes placas en documentos del mismo caso',
            'condition': 'numero_placas_diferentes > 1',
            'action': 'ADD fraud_score += 0.8',
            'evidence_from': 'Caso MODA YKT - Placas diferentes en carta porte',
            'confidence': 0.85
        })

        # REGLAS DE CORRELACIÓN

        # Regla 5: Reporte temprano + sin evidencia física (Caso Castro)
        rules['correlation_rules'].append({
            'id': 'RULE_005',
            'name': 'reporte_temprano_sin_evidencia',
            'description': 'Reporte muy temprano sin evidencia física esperada',
            'condition': 'dias_desde_poliza < 30 AND NOT olor_quemado AND tipo_siniestro = voltaje',
            'action': 'ADD fraud_score += 0.6',
            'evidence_from': 'Caso Castro - 14 días + sin olor',
            'confidence': 0.7
        })

        return rules

    def compile_rules_to_code(self, rules):
        """Compilar reglas a código Python ejecutable"""

        code = '''
class CompiledFraudRules:
    """
    Reglas heurísticas compiladas basadas en casos reales de HDI
    """

    def apply_rules(self, case_features, verification_results):
        fraud_score = 0.0
        triggered_rules = []

        # REGLAS DETERMINANTES
        '''

        # Generar código para reglas determinantes
        for rule in rules['determinant_rules']:
            condition = self.translate_condition_to_python(rule['condition'])
            action = self.translate_action_to_python(rule['action'])

            code += f'''
        # {rule['name']} - {rule['evidence_from']}
        if {condition}:
            {action}
            triggered_rules.append({{
                'rule_id': '{rule['id']}',
                'name': '{rule['name']}',
                'confidence': {rule['confidence']},
                'evidence': '{rule['evidence_from']}'
            }})
            '''

        # Generar código para reglas de alta probabilidad
        code += '\n        # REGLAS DE ALTA PROBABILIDAD\n'

        for rule in rules['high_probability_rules']:
            condition = self.translate_condition_to_python(rule['condition'])
            action = self.translate_action_to_python(rule['action'])

            code += f'''
        # {rule['name']} - {rule['evidence_from']}
        if {condition}:
            {action}
            triggered_rules.append({{
                'rule_id': '{rule['id']}',
                'name': '{rule['name']}',
                'confidence': {rule['confidence']},
                'evidence': '{rule['evidence_from']}'
            }})
            '''

        code += '''

        return {
            'fraud_score': min(fraud_score, 1.0),
            'triggered_rules': triggered_rules
        }
        '''

        return code
```

---

## 🔬 VALIDACIÓN CON CASOS REALES

### Protocolo de Validación Cruzada

```python
class RealCaseValidator:
    """
    Validador que utiliza los casos reales como ground truth
    """

    def __init__(self):
        self.real_cases = {
            '20250000004129': {'verdict': 'CON_TENTATIVA', 'confidence': 0.95},
            '20250000002494': {'verdict': 'CON_TENTATIVA', 'confidence': 0.98},
            '20240000001361': {'verdict': 'CON_TENTATIVA', 'confidence': 0.92},
            '20250000003818': {'verdict': 'SIN_TENTATIVA', 'confidence': 0.85}
        }

    def validate_models(self, trained_models):
        """Validar modelos contra casos reales conocidos"""

        results = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'case_details': []
        }

        correct_predictions = 0
        total_cases = len(self.real_cases)

        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for case_id, expected in self.real_cases.items():
            # Obtener predicción del modelo
            prediction = trained_models.predict(case_id)

            # Verificar si la predicción es correcta
            is_correct = prediction['verdict'] == expected['verdict']
            if is_correct:
                correct_predictions += 1

            # Calcular métricas detalladas
            if expected['verdict'] == 'CON_TENTATIVA' and prediction['verdict'] == 'CON_TENTATIVA':
                true_positives += 1
            elif expected['verdict'] == 'SIN_TENTATIVA' and prediction['verdict'] == 'CON_TENTATIVA':
                false_positives += 1
            elif expected['verdict'] == 'CON_TENTATIVA' and prediction['verdict'] == 'SIN_TENTATIVA':
                false_negatives += 1

            results['case_details'].append({
                'case_id': case_id,
                'expected': expected['verdict'],
                'predicted': prediction['verdict'],
                'confidence': prediction['confidence'],
                'correct': is_correct
            })

        # Calcular métricas finales
        results['accuracy'] = correct_predictions / total_cases
        results['precision'] = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        results['recall'] = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

        if results['precision'] + results['recall'] > 0:
            results['f1_score'] = 2 * (results['precision'] * results['recall']) / (results['precision'] + results['recall'])

        return results
```

---

## 🚀 PLAN DE IMPLEMENTACIÓN

### Fase 1: Preparación de Datos (1 semana)

1. **Conversión de Reportes TXT a JSON Estructurado**
   - Parser para reportes de HDI
   - Extractor de secciones estándar
   - Validador de estructura

2. **Creación de Dataset de Verificaciones**
   - Logs de verificaciones manuales
   - Resultados de APIs cuando estén disponibles
   - Correlación con verdics finales

### Fase 2: Entrenamiento Base (2 semanas)

1. **Modelos Especializados por Tipo de Siniestro**
   - Clasificador para daños eléctricos
   - Clasificador para robos en tránsito
   - Clasificador genérico

2. **Sistema de Reglas Heurísticas**
   - Compilador de reglas basadas en casos reales
   - Motor de ejecución de reglas
   - Sistema de pesos dinámicos

### Fase 3: Integración con Verificaciones (1 semana)

1. **Pipeline Híbrido**
   - Generador de checklists inteligente
   - Procesador de verificaciones manuales
   - Actualización de scores con datos verificados

2. **Sistema de Aprendizaje Continuo**
   - Captura de feedback de analistas
   - Reentrenamiento incremental
   - Mejora de reglas automática

### Fase 4: Validación y Ajuste (1 semana)

1. **Validación con Casos Reales**
   - Testing con los 4 casos base
   - Ajuste de umbrales
   - Calibración de confianza

2. **Testing de Integración**
   - Pipeline completo end-to-end
   - Performance y latencia
   - Manejo de errores

---

## 📊 MÉTRICAS ESPERADAS

### Objetivos de Performance

| Métrica | Meta | Basado en |
|---------|------|-----------|
| **Precisión Global** | 95% | Casos reales analizados |
| **Recall para Fraude** | 100% | Casos CON TENTATIVA |
| **Falsos Positivos** | < 5% | Casos SIN TENTATIVA |
| **Tiempo de Análisis** | < 10 min | Con verificaciones manuales |

### KPIs por Tipo de Siniestro

**Daños Eléctricos:**
- Detección de diagnósticos imposibles: 100%
- Verificación de evidencia física: 90%

**Robos en Tránsito:**
- Detección de documentos apócrifos: 95%
- Identificación de inconsistencias vehiculares: 90%

---

## 🔮 EVOLUCIÓN FUTURA

### Cuando APIs Estén Disponibles

1. **Migración Automática**
   - Cambio de modo manual a automático
   - Mantenimiento de fallback manual
   - Validación cruzada entre APIs y manual

2. **Aprendizaje Acelerado**
   - Mayor volumen de datos
   - Validaciones en tiempo real
   - Patrones más complejos

### Expansión del Sistema

1. **Nuevos Tipos de Siniestro**
   - Adaptación a otros productos de seguro
   - Generalización de patrones
   - Especialización por mercado

2. **Análisis Predictivo**
   - Identificación de riesgos antes del siniestro
   - Patrones de comportamiento sospechoso
   - Recomendaciones preventivas

---

*Este documento establece las bases técnicas para implementar un sistema de ML robusto y efectivo, basado en evidencia real y diseñado para evolucionar con el negocio.*

**Próximos Pasos:**
1. Implementar parser de reportes TXT
2. Crear primer modelo con 4 casos base
3. Integrar con sistema de verificaciones manuales
4. Expandir con los 60 reportes completos