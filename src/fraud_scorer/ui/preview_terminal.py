"""
Interfaz de terminal para preview de clasificación de documentos.

Esta UI utiliza la librería Rich para mostrar tablas, paneles, árboles
y estadísticas de forma clara y sin persistir datos.
"""

from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich.markdown import Markdown
from rich import box
from rich.columns import Columns


class PreviewTerminalUI:
    """UI de terminal enriquecida con Rich."""

    def __init__(self, console: Console):
        # Consola Rich a utilizar para el render
        self.console = console

    def show_header(self, input_folder: Path):
        """Muestra encabezado con carpeta y modo de ejecución."""

        header = Panel(
            f"""
[bold cyan]📂 Carpeta analizada:[/bold cyan] {input_folder}
[bold cyan]🕐 Inicio:[/bold cyan] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
[bold cyan]🔧 Modo:[/bold cyan] Preview sin caché (no se guardan datos)
            """,
            title="[bold]Preview de Clasificación de Documentos[/bold]",
            subtitle="[dim]Sistema de testing sin persistencia[/dim]",
            border_style="cyan",
            padding=(1, 2),
        )

        self.console.print(header)

    def show_classification_tree(self, results: List[Dict[str, Any]]):
        """Muestra un árbol agrupando documentos por tipo detectado."""

        tree = Tree("📁 Documentos Clasificados", style="bold cyan")

        # Agrupar por tipo
        by_type: Dict[str, List[Dict[str, Any]]] = {}
        for result in results:
            by_type.setdefault(result["document_type"], []).append(result)

        # Crear nodos por tipo
        for doc_type, docs in sorted(by_type.items()):
            type_node = tree.add(
                f"[green]{doc_type}[/green] ({len(docs)} archivos)", style="bold"
            )
            # Mostrar hasta 5 ejemplos por tipo
            for doc in docs[:5]:
                conf_color = self._get_confidence_color(doc["confidence"])
                type_node.add(
                    f"📄 {doc['filename'][:30]} "
                    f"[{conf_color}]({doc['confidence']:.0%})[/{conf_color}]"
                )
            if len(docs) > 5:
                type_node.add(f"[dim]... y {len(docs) - 5} más[/dim]")

        self.console.print("\n")
        self.console.print(tree)

    def show_confidence_distribution(self, results: List[Dict[str, Any]]):
        """Muestra distribución simple de niveles de confianza."""

        high = sum(1 for r in results if r["confidence"] >= 0.8)
        medium = sum(1 for r in results if 0.6 <= r["confidence"] < 0.8)
        low = sum(1 for r in results if r["confidence"] < 0.6)
        total = max(len(results), 1)

        # Barras aproximadas por proporción (hasta 30 bloques)
        bar = lambda n: "█" * int((n / total) * 30)
        panel_content = (
            f"[green]Alta (≥80%):[/green]   {bar(high)} {high} ({high/total:.0%})\n"
            f"[yellow]Media (60-79%):[/yellow] {bar(medium)} {medium} ({medium/total:.0%})\n"
            f"[red]Baja (<60%):[/red]    {bar(low)} {low} ({low/total:.0%})\n"
        )

        panel = Panel(
            panel_content,
            title="[bold]Distribución de Confianza[/bold]",
            border_style="blue",
            padding=(1, 2),
        )

        self.console.print("\n")
        self.console.print(panel)

    def show_method_stats(self, results: List[Dict[str, Any]]):
        """Muestra estadísticas por método (heurística vs LLM)."""

        heuristic = sum(1 for r in results if r["method"] == "heuristic")
        llm = sum(1 for r in results if r["method"] == "llm")
        total = max(len(results), 1)

        col1 = Panel(
            f"""[bold cyan]📏 Heurística[/bold cyan]

Archivos: {heuristic}
Porcentaje: {heuristic/total:.0%}
Promedio confianza: {self._avg_confidence(results, 'heuristic'):.0%}
""",
            border_style="cyan",
            padding=(1, 2),
        )

        col2 = Panel(
            f"""[bold magenta]🤖 LLM[/bold magenta]

Archivos: {llm}
Porcentaje: {llm/total:.0%}
Promedio confianza: {self._avg_confidence(results, 'llm'):.0%}
""",
            border_style="magenta",
            padding=(1, 2),
        )

        self.console.print("\n")
        self.console.print(Columns([col1, col2]))

    def show_ocr_requirements(self, results: List[Dict[str, Any]]):
        """Muestra un cuadro con requerimientos de OCR estimados."""

        need_ocr = [r for r in results if r["needs_ocr"]]
        no_ocr = [r for r in results if not r["needs_ocr"]]

        table = Table(
            title="🔍 Requisitos de OCR en Producción",
            show_header=True,
            header_style="bold",
            box=box.ROUNDED,
        )

        table.add_column("Categoría", style="cyan")
        table.add_column("Cantidad", justify="center")
        table.add_column("Porcentaje", justify="center")
        table.add_column("Ejemplos", no_wrap=False)

        if need_ocr:
            examples = ", ".join([r["filename"][:20] for r in need_ocr[:3]])
            if len(need_ocr) > 3:
                examples += f" (+{len(need_ocr)-3} más)"
            table.add_row(
                "✅ Necesitan OCR",
                str(len(need_ocr)),
                f"{len(need_ocr)/max(len(results),1):.0%}",
                examples,
            )

        if no_ocr:
            examples = ", ".join([r["filename"][:20] for r in no_ocr[:3]])
            if len(no_ocr) > 3:
                examples += f" (+{len(no_ocr)-3} más)"
            table.add_row(
                "❌ No necesitan OCR",
                str(len(no_ocr)),
                f"{len(no_ocr)/max(len(results),1):.0%}",
                examples,
            )

        self.console.print("\n")
        self.console.print(table)

    def _get_confidence_color(self, confidence: float) -> str:
        """Retorna color según el nivel de confianza."""
        if confidence >= 0.8:
            return "green"
        if confidence >= 0.6:
            return "yellow"
        return "red"

    def _avg_confidence(self, results: List[Dict[str, Any]], method: str) -> float:
        """Calcula confianza promedio por método indicado."""
        filtered = [r["confidence"] for r in results if r["method"] == method]
        return sum(filtered) / len(filtered) if filtered else 0.0
