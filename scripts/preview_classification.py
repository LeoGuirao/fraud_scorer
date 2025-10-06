#!/usr/bin/env python3
"""
Sistema de Preview de Clasificación de Documentos (sin persistencia).

- No genera caché ni archivos temporales por defecto.
- Flujo LLM-first (GPT económico) con fallback heurístico ante errores.
- UI de terminal con Rich para presentar resultados y detalles.

Notas:
- La exportación a JSON es opcional mediante parámetro CLI y es la única
  operación que escribe en disco.
- Todos los comentarios y docstrings están en español.
"""

from __future__ import annotations

import sys
import asyncio
import json
from pathlib import Path
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown

# Intentar cargar YAML si está disponible (config opcional)
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - dependencia opcional
    yaml = None  # type: ignore

# Incluir "src" en sys.path para imports del paquete principal
CURRENT_DIR = Path(__file__).parent
sys.path.append(str(CURRENT_DIR.parent / "src"))

from fraud_scorer.processors.preview.preview_classifier import (
    DocumentPreviewClassifier,
)
from fraud_scorer.ui.preview_terminal import PreviewTerminalUI
from fraud_scorer.settings import SUPPORTED_EXTENSIONS, ExtractionConfig
from fraud_scorer.settings import ExtractionRoute  # Enum de rutas


# Consola y app Typer
console = Console()
app = typer.Typer(
    name="preview-classifier",
    help="Preview de clasificación de documentos sin generar caché",
    add_completion=False,
)


def _load_dotenv_best_effort() -> None:
    """Carga variables desde .env si es posible.

    1) Intenta con python-dotenv si está instalado.
    2) Si no, aplica un parser simple de KEY=VALUE sobre el .env en la raíz.
    """
    # Ruta estándar del repo (raíz)
    env_path = CURRENT_DIR.parent / ".env"

    # Opción 1: usar python-dotenv
    try:  # pragma: no cover
        from dotenv import load_dotenv, find_dotenv  # type: ignore
        # Intentar encontrar automáticamente; si no, usar env_path
        found = find_dotenv(usecwd=True)
        if found:
            load_dotenv(found, override=False)
            return
        if env_path.exists():
            load_dotenv(env_path, override=False)
            return
    except Exception:
        pass

    # Opción 2: parseo simple si no está dotenv
    try:
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                os.environ.setdefault(k, v)
    except Exception:
        # Silencioso: no es crítico si .env no puede cargarse
        pass


def load_preview_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Carga configuración desde .preview_config.yaml si existe.

    - Si PyYAML no está disponible o el archivo no existe, retorna defaults.
    - La ruta por defecto es el raíz del repo (padre del directorio scripts).
    """

    defaults: Dict[str, Any] = {
        "preview": {
            "llm_model": "gpt-5",
            "confidence_threshold": 0.6,
            "max_sample_chars": 2000,
            "supported_extensions": list(SUPPORTED_EXTENSIONS),
            "ui": {
                "show_samples_by_default": False,
                "max_files_per_type_in_tree": 5,
                "enable_colors": True,
            },
            "limits": {
                "max_files": 1000,
                "max_file_size_mb": 50,
            },
        }
    }

    if yaml is None:
        return defaults

    try:
        path = config_path or (CURRENT_DIR.parent / ".preview_config.yaml")
        if path.exists():
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            # Merge superficial: solo primer nivel bajo "preview"
            if isinstance(data, dict) and "preview" in data:
                defaults["preview"].update(data["preview"])
    except Exception as e:  # pragma: no cover - robustez
        console.print(f"[yellow]⚠️ No se pudo cargar config YAML: {e}[/yellow]")

    return defaults


class PreviewSession:
    """Sesión de clasificación en memoria (sin persistencia)."""

    def __init__(self, input_folder: Path, config: Dict[str, Any], api_key: Optional[str] = None):
        self.input_folder = input_folder
        self.config = config.get("preview", {})

        # Inicializar clasificador con modelo de la configuración
        model_name = self.config.get("llm_model") or "gpt-5-mini"
        self.classifier = DocumentPreviewClassifier(model_name, api_key=api_key)

        # UI de terminal
        self.ui = PreviewTerminalUI(console)
        
        # Resultados acumulados de la sesión
        self.results: List[Dict[str, Any]] = []
        self.start_time = datetime.now()
        # Estado de soporte interactivo (questionary)
        self._interactive_supported: bool = True

    async def run(
        self,
        use_llm: bool = True,
        show_samples: bool = False,
        interactive: bool = True,
        use_vision: bool = True,
    ) -> None:
        """Ejecuta el flujo de análisis de la carpeta dada."""

        # Cabecera
        self.ui.show_header(self.input_folder)

        # Descubrir archivos según filtros y límites
        files = self._discover_files()
        if not files:
            console.print("[red]❌ No se encontraron archivos soportados[/red]")
            return

        console.print(f"\n📁 Encontrados [cyan]{len(files)}[/cyan] archivos para analizar")

        # En modo interactivo se pueden ajustar flags
        if interactive:
            config = await self._interactive_config()
            use_llm = config["use_llm"]
            show_samples = config["show_samples"]
            if not self._interactive_supported:
                # Si no hay questionary, desactivar modo interactivo y continuar
                interactive = False

        # Aviso si LLM solicitado pero no disponible
        if use_llm and getattr(self.classifier, "client", None) is None:
            console.print(
                "[yellow]⚠️  LLM no disponible: define OPENAI_API_KEY o usa --api-key para habilitarlo. Se usará heurística.[/yellow]"
            )

        # Progreso de análisis
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("[cyan]Analizando documentos...", total=len(files))

            for idx, file_path in enumerate(files, 1):
                progress.update(task, description=f"[cyan]Analizando {file_path.name}...")

                # Extraer muestra de texto sin OCR
                sample_text = await self._extract_sample(file_path)

                # Clasificar (LLM con visión por defecto). Se pasa la ruta del
                # documento al clasificador para que decida el mejor método.
                doc_type, confidence, reasons, method = await self.classifier.classify(
                    sample_text=sample_text,
                    filename=file_path.name,
                    use_llm=use_llm,
                    use_vision=use_vision,
                    document_path=(file_path if use_vision else None),
                )

                # Determinar si requeriría OCR en producción
                needs_ocr = self._would_need_ocr(file_path, doc_type, sample_text)

                self.results.append(
                    {
                        "index": idx,
                        "filename": file_path.name,
                        "path": str(file_path),
                        "size": file_path.stat().st_size,
                        "document_type": doc_type,
                        "confidence": confidence,
                        "reasons": reasons,
                        "method": method,
                        "needs_ocr": needs_ocr,
                        "sample_text": sample_text[:500] if sample_text else None,
                    }
                )
                progress.advance(task)

        # Tabla de resultados, detalles y resumen
        self._show_results_table()
        if show_samples or interactive:
            await self._show_classification_details()
        self._show_summary()

    def _discover_files(self) -> List[Path]:
        """Descubre archivos soportados aplicando límites de tamaño y cantidad."""

        # Extensiones soportadas: YAML → settings → fallback local
        cfg_exts = set(self.config.get("supported_extensions") or [])
        exts = cfg_exts or set(SUPPORTED_EXTENSIONS) or {
            ".pdf",
            ".jpg",
            ".jpeg",
            ".png",
            ".docx",
            ".xlsx",
            ".csv",
            ".txt",
        }

        # Límites
        limits = self.config.get("limits", {})
        max_files = int(limits.get("max_files", 1000))
        max_file_size_mb = int(limits.get("max_file_size_mb", 50))
        max_file_size_bytes = max_file_size_mb * 1024 * 1024

        files: List[Path] = []
        for ext in exts:
            files.extend(self.input_folder.glob(f"*{ext}"))
            files.extend(self.input_folder.glob(f"*{ext.upper()}"))

        # Filtrar por tamaño y truncar por cantidad
        files = [f for f in sorted(files) if f.stat().st_size <= max_file_size_bytes]
        if len(files) > max_files:
            files = files[:max_files]

        return files

    async def _interactive_config(self) -> Dict[str, Any]:
        """Recoge opciones de ejecución en modo interactivo."""
        try:
            import questionary  # type: ignore
        except ModuleNotFoundError:
            self._interactive_supported = False
            console.print(
                "[yellow]⚠️  'questionary' no está instalado. Continuando en modo no interactivo.[/yellow]"
            )
            # Defaults seguros
            return {"use_llm": True, "show_samples": False}

        console.print("\n⚙️  [bold]Configuración del Preview[/bold]\n")

        config: Dict[str, Any] = {}
        config["use_llm"] = await asyncio.to_thread(
            lambda: questionary.confirm(
                "¿Usar LLM para clasificación?", default=True
            ).ask()
        )
        config["show_samples"] = await asyncio.to_thread(
            lambda: questionary.confirm(
                "¿Mostrar muestras de texto extraído?", default=False
            ).ask()
        )

        return config

    async def _extract_sample(self, file_path: Path) -> str:
        """Extrae muestra de texto SIN OCR (según extensión)."""

        max_chars = int(self.config.get("max_sample_chars", 2000))
        ext = file_path.suffix.lower()

        try:
            if ext == ".txt":
                return file_path.read_text(encoding="utf-8", errors="ignore")[:max_chars]

            if ext == ".pdf":
                # 1) Intentar PyPDF2
                try:
                    import PyPDF2  # type: ignore
                    text = ""
                    with file_path.open("rb") as f:
                        reader = PyPDF2.PdfReader(f)
                        for page in reader.pages[:2]:
                            extracted = page.extract_text() or ""
                            text += extracted
                            if len(text) >= max_chars:
                                break
                    if len(text.strip()) >= 50:
                        return text[:max_chars]
                except Exception:
                    text = ""

                # 2) Fallback: pdfminer.six (si está instalado)
                try:
                    from pdfminer.high_level import extract_text  # type: ignore
                    text2 = extract_text(str(file_path), maxpages=2) or ""
                    if text2:
                        return text2[:max_chars]
                except Exception:
                    # No disponible o falló → sin texto
                    pass

            if ext in {".docx"}:
                from docx import Document  # type: ignore

                doc = Document(str(file_path))
                text = "\n".join(p.text for p in doc.paragraphs[:20])
                return text[:max_chars]

            if ext in {".xlsx", ".csv"}:
                import pandas as pd  # type: ignore

                if ext == ".csv":
                    # Parametro deprecated reemplazado por on_bad_lines
                    df = pd.read_csv(file_path, nrows=10, on_bad_lines="skip")
                else:
                    df = pd.read_excel(file_path, nrows=10)
                return df.to_string()[:max_chars]

            if ext in {".jpg", ".jpeg", ".png"}:
                # Imágenes: sin OCR/vision en extracción de texto; el flujo de visión
                # (si se habilita) ocurre en la etapa de clasificación, no aquí.
                return ""
        except Exception as e:  # pragma: no cover - robustez de lectura
            console.print(
                f"[yellow]⚠️  No se pudo extraer texto de {file_path.name}: {e}[/yellow]"
            )

        return ""

    def _would_need_ocr(self, file_path: Path, doc_type: str, sample_text: str) -> bool:
        """Heurística para estimar si requeriría OCR en producción.

        - Imágenes: siempre sí.
        - PDFs con muestra casi vacía: probablemente sí.
        - Según rutas definidas en settings: OCR_TEXT → sí.
        """

        ext = file_path.suffix.lower()
        if ext in {".jpg", ".jpeg", ".png"}:
            return True

        if ext == ".pdf" and len((sample_text or "").strip()) < 100:
            return True

        try:
            routes = ExtractionConfig.DOCUMENT_EXTRACTION_ROUTES
            route = routes.get(doc_type)
            if route == ExtractionRoute.OCR_TEXT:
                return True
        except Exception:
            # Ante cualquier error en config, ser conservadores
            return False

        return False

    def _show_results_table(self) -> None:
        """Muestra una tabla con los resultados de clasificación."""

        table = Table(
            title="📋 Resultados de Clasificación",
            show_header=True,
            header_style="bold magenta",
            title_style="bold",
            box=None,
            padding=(0, 1),
        )

        table.add_column("#", style="dim", width=4)
        table.add_column("Archivo", style="cyan", no_wrap=False)
        table.add_column("Tipo Detectado", style="green")
        table.add_column("Confianza", justify="center")
        table.add_column("Método", justify="center")
        table.add_column("Necesita OCR", justify="center")

        for result in self.results:
            conf = result["confidence"]
            conf_color = "green" if conf >= 0.8 else ("yellow" if conf >= 0.6 else "red")
            method_icon = "🤖" if result["method"] == "llm" else "📏"
            ocr_icon = "✅" if result["needs_ocr"] else "❌"

            table.add_row(
                str(result["index"]),
                result["filename"][:40],
                result["document_type"],
                f"[{conf_color}]{conf:.1%}[/{conf_color}]",
                method_icon,
                ocr_icon,
            )

        console.print("\n")
        console.print(table)

    async def _show_classification_details(self) -> None:
        """Muestra paneles con detalles de clasificación de documentos."""
        try:
            import questionary  # type: ignore
        except ModuleNotFoundError:
            console.print(
                "[yellow]⚠️  'questionary' no está instalado. Se omite el detalle interactivo.[/yellow]"
            )
            return

        console.print("\n📝 [bold]Detalles de Clasificación[/bold]\n")
        choice = await asyncio.to_thread(
            lambda: questionary.select(
                "¿Qué detalles quieres ver?",
                choices=[
                    "Ver todos los archivos",
                    "Seleccionar archivos específicos",
                    "Ver solo clasificaciones con baja confianza (< 60%)",
                    "Ver solo clasificaciones por LLM",
                    "Saltar detalles",
                ],
            ).ask()
        )

        if choice == "Saltar detalles":
            return

        if choice == "Ver todos los archivos":
            to_show = self.results
        elif choice == "Seleccionar archivos específicos":
            choices = [f"{r['index']}. {r['filename']}" for r in self.results]
            selected = await asyncio.to_thread(
                lambda: questionary.checkbox("Selecciona archivos:", choices=choices).ask()
            )
            selected_indices = {int(s.split(".")[0]) for s in (selected or [])}
            to_show = [r for r in self.results if r["index"] in selected_indices]
        elif choice == "Ver solo clasificaciones con baja confianza (< 60%)":
            to_show = [r for r in self.results if r["confidence"] < 0.6]
        else:
            to_show = [r for r in self.results if r["method"] == "llm"]

        for result in to_show:
            self._show_single_detail(result)

    def _show_single_detail(self, result: Dict[str, Any]) -> None:
        """Muestra detalle enriquecido de un resultado individual."""

        content = f"""
**Archivo:** {result['filename']}
**Tipo detectado:** {result['document_type']}
**Confianza:** {result['confidence']:.1%}
**Método:** {'🤖 LLM' if result['method'] == 'llm' else '📏 Heurística'}
**Necesita OCR:** {'✅ Sí' if result['needs_ocr'] else '❌ No'}
**Tamaño:** {result['size'] / 1024:.1f} KB

**📍 Razones de clasificación:**
"""

        for reason in result.get("reasons", []):
            content += f"  • {reason}\n"

        sample: Optional[str] = result.get("sample_text")
        if sample and len(sample) > 10:
            content += (
                "\n**📄 Muestra de texto (primeros 200 chars):**\n```\n"
                + sample[:200]
                + "...\n```"
            )

        panel = Panel(
            Markdown(content),
            title=f"[bold]#{result['index']} - {result['document_type']}[/bold]",
            border_style="blue",
            padding=(1, 2),
        )

        console.print(panel)
        console.print("")

    def _show_summary(self) -> None:
        """Muestra un resumen final de la sesión de preview."""

        total = len(self.results)
        by_type: Dict[str, int] = {}
        by_method = {"heuristic": 0, "llm": 0}
        avg_confidence = 0.0
        need_ocr = 0

        for r in self.results:
            by_type[r["document_type"]] = by_type.get(r["document_type"], 0) + 1
            by_method[r["method"]] += 1
            avg_confidence += r["confidence"]
            if r["needs_ocr"]:
                need_ocr += 1

        avg_confidence = avg_confidence / total if total else 0.0
        elapsed = (datetime.now() - self.start_time).total_seconds()

        summary = f"""
## 📊 Resumen de Clasificación

**Total de archivos:** {total}
**Tiempo de análisis:** {elapsed:.1f} segundos
**Confianza promedio:** {avg_confidence:.1%}

### 📁 Distribución por tipo:
"""

        for doc_type, count in sorted(by_type.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total) * 100 if total else 0.0
            summary += f"  • **{doc_type}:** {count} ({percentage:.1f}%)\n"

        summary += f"""

### 🔧 Métodos utilizados:
  • **Heurística:** {by_method['heuristic']} ({(by_method['heuristic']/total)*100 if total else 0:.1f}%)
  • **LLM:** {by_method['llm']} ({(by_method['llm']/total)*100 if total else 0:.1f}%)

### 🔍 Requisitos de OCR:
  • **Necesitan OCR:** {need_ocr} archivos ({(need_ocr/total)*100 if total else 0:.1f}%)
  • **No necesitan OCR:** {total - need_ocr} archivos ({(((total-need_ocr)/total)*100) if total else 0:.1f}%)
"""

        console.print("\n")
        console.print(
            Panel(
                Markdown(summary),
                title="[bold green]✅ Análisis Completado[/bold green]",
                border_style="green",
                padding=(1, 2),
            )
        )


@app.command()
def preview(
    folder: Path = typer.Argument(
        ..., help="Carpeta con documentos a clasificar", exists=True, file_okay=False, dir_okay=True, resolve_path=True
    ),
    no_llm: bool = typer.Option(False, "--no-llm", help="No usar LLM, solo heurística"),
    show_samples: bool = typer.Option(False, "--show-samples", "-s", help="Mostrar muestras de texto extraído"),
    non_interactive: bool = typer.Option(False, "--non-interactive", "-n", help="Modo no interactivo (sin prompts)"),
    export_json: Optional[Path] = typer.Option(None, "--export-json", "-e", help="Exportar resultados a archivo JSON"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="OpenAI API Key (sino usa OPENAI_API_KEY)"),
    vision: bool = typer.Option(True, "--use-vision", help="Usar clasificación por visión del LLM para analizar PDFs/Imágenes"),
    config_path: Optional[Path] = typer.Option(None, "--config", help="Ruta a .preview_config.yaml personalizado"),
):
    """CLI de preview de clasificación sin caché ni persistencia."""

    console.print(
        """
╔══════════════════════════════════════════════════════════╗
║     🔍 PREVIEW DE CLASIFICACIÓN DE DOCUMENTOS 🔍        ║
║            Sistema de Testing Sin Caché                  ║
╚══════════════════════════════════════════════════════════╝
""",
        style="bold cyan",
    )

    # Cargar .env para habilitar OPENAI_API_KEY si existe
    _load_dotenv_best_effort()

    # Cargar configuración
    cfg = load_preview_config(config_path)

    # Crear sesión
    session = PreviewSession(folder, cfg, api_key=api_key)

    # Ejecutar
    asyncio.run(
        session.run(
            use_llm=not no_llm,
            show_samples=show_samples,
            interactive=not non_interactive,
            use_vision=vision,
        )
    )

    # Exportar si se solicitó (única escritura a disco)
    if export_json:
        with export_json.open("w", encoding="utf-8") as f:
            json.dump(session.results, f, indent=2, ensure_ascii=False)
        console.print(f"\n💾 Resultados exportados a: {export_json}")

    console.print(
        "\n[bold green]✨ Preview completado. No se generó caché ni datos persistentes.[/bold green]"
    )


if __name__ == "__main__":
    app()
