# src/fraud_scorer/replay/replay_ui.py

"""
Interfaz interactiva para el sistema de replay
"""
import os
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.layout import Layout
from rich.text import Text
from rich import box
import asyncio
import logging
from getpass import getpass  # <-- Captura oculta de API key

console = Console()
logger = logging.getLogger(__name__)

class ReplayUI:
    """
    Interfaz de usuario interactiva para replay de casos
    """
    
    def __init__(self, cache_manager, system):
        self.cache_manager = cache_manager
        self.system = system
        self.selected_case = None
        
    def clear_screen(self):
        """Limpia la pantalla"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def show_header(self):
        """Muestra el header del sistema"""
        self.clear_screen()
        console.print(Panel.fit(
            "[bold cyan]🔄 FRAUD SCORER - REPLAY SYSTEM[/bold cyan]\n"
            "[dim]Sistema de Re-procesamiento con Cache OCR[/dim]",
            border_style="cyan"
        ))
        console.print()
    
    def show_main_menu(self) -> str:
        """Muestra el menú principal"""
        self.show_header()
        
        # Obtener estadísticas del cache
        stats = self.cache_manager.get_cache_stats()
        
        # Panel de estadísticas
        stats_text = (
            f"[cyan]📊 Estadísticas del Cache[/cyan]\n"
            f"├─ Casos en cache: [bold]{stats['total_cases']}[/bold]\n"
            f"├─ Archivos cacheados: [bold]{stats['total_cached_files']}[/bold]\n"
            f"└─ Tamaño total: [bold]{stats['cache_size_mb']} MB[/bold]"
        )
        console.print(Panel(stats_text, box=box.ROUNDED))
        console.print()
        
        # Opciones del menú
        console.print("[bold]Opciones disponibles:[/bold]")
        console.print("  [1] 📋 Ver casos disponibles para replay")
        console.print("  [2] 🔄 Replay de un caso específico")
        console.print("  [3] 🗑️  Limpiar cache de un caso")
        console.print("  [4] 📈 Ver estadísticas detalladas")
        console.print("  [5] ❌ Salir")
        console.print()
        
        choice = Prompt.ask(
            "[bold yellow]Seleccione una opción[/bold yellow]",
            choices=["1", "2", "3", "4", "5"],
            default="1"
        )
        
        return choice
    
    def show_cases_list(self) -> Optional[str]:
        """Muestra la lista de casos disponibles"""
        self.show_header()
        
        cases = self.cache_manager.list_cached_cases()
        
        if not cases:
            console.print("[yellow]⚠️ No hay casos en cache[/yellow]")
            console.print()
            Prompt.ask("Presione Enter para continuar")
            return None
        
        # Crear tabla de casos
        table = Table(
            title="[bold cyan]Casos Disponibles para Replay[/bold cyan]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta"
        )
        
        table.add_column("#", style="cyan", width=4)
        table.add_column("Case ID", style="green")
        table.add_column("Título", style="white")
        table.add_column("Documentos", justify="center", style="yellow")
        table.add_column("Procesado", style="blue")
        
        for idx, case in enumerate(cases, 1):
            # Formatear fecha
            proc_date = datetime.fromisoformat(case['processed_at'])
            date_str = proc_date.strftime("%d/%m/%Y %H:%M")
            
            table.add_row(
                str(idx),
                case['case_id'],
                case['case_title'][:40] + "..." if len(case['case_title']) > 40 else case['case_title'],
                str(case['total_documents']),
                date_str
            )
        
        console.print(table)
        console.print()
        
        # Selección
        console.print("[bold]Opciones:[/bold]")
        console.print("  • Ingrese el número del caso para seleccionarlo")
        console.print("  • Ingrese 'v' para volver al menú principal")
        console.print("  • Ingrese 'q' para salir")
        console.print()
        
        while True:
            choice = Prompt.ask("[bold yellow]Seleccione[/bold yellow]")
            
            if choice.lower() == 'v':
                return 'back'
            elif choice.lower() == 'q':
                return 'quit'
            
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(cases):
                    return cases[idx]['case_id']
                else:
                    console.print("[red]Número inválido[/red]")
            except ValueError:
                console.print("[red]Entrada inválida[/red]")
    
    def show_case_details(self, case_id: str) -> Dict[str, Any]:
        """Muestra los detalles de un caso y opciones de replay"""
        self.show_header()
        
        # Obtener información del caso
        case_index = self.cache_manager.get_case_index(case_id)
        
        if not case_index:
            console.print(f"[red]❌ No se encontró información del caso {case_id}[/red]")
            Prompt.ask("Presione Enter para continuar")
            return {'action': 'back'}
        
        # Panel de información del caso
        info_text = (
            f"[bold cyan]📁 Caso: {case_id}[/bold cyan]\n"
            f"[white]Título:[/white] {case_index['case_title']}\n"
            f"[white]Documentos:[/white] {case_index['total_documents']}\n"
            f"[white]Procesado:[/white] {case_index['processed_at']}\n"
            f"[white]Carpeta:[/white] {case_index.get('folder_path', 'N/A')}"
        )
        console.print(Panel(info_text, box=box.DOUBLE))
        console.print()
        
        # Opciones de procesamiento con mejor guía
        console.print("[bold]Opciones de Re-procesamiento:[/bold]")
        console.print("[dim]Presione Enter para usar valores por defecto[/dim]")
        console.print()
        
        options: Dict[str, Any] = {}
        
        # Modo de procesamiento
        console.print("[cyan]1. Modo de procesamiento:[/cyan]")
        console.print("   [dim]IA usa la familia GPT-5 para extracción inteligente[/dim]")
        use_ai = Confirm.ask("   ¿Usar sistema de IA?", default=True)
        options['use_ai'] = use_ai
        console.print()
        
        # Análisis por documento
        if use_ai:
            console.print("[cyan]2. Análisis detallado:[/cyan]")
            console.print("   [dim]Analiza cada documento individualmente (más lento)[/dim]")
            per_doc = Confirm.ask("   ¿Realizar análisis por documento?", default=False)
            options['per_doc'] = per_doc
            console.print()
            
            # Modelo de IA
            console.print("[cyan]3. Configuración de IA:[/cyan]")
            console.print("   [dim]gpt-5-mini equilibra costo y velocidad[/dim]")
            console.print("   [dim]gpt-5 ofrece máxima precisión; gpt-5-thinking profundiza si el caso lo requiere[/dim]")
            model = Prompt.ask(
                "   Modelo a usar",
                default="gpt-5",
                choices=["gpt-5", "gpt-5-thinking", "gpt-5-mini", "gpt-5-vision"]
            )
            options['model'] = model

            console.print("   [dim]0.1 = preciso, 0.7 = creativo[/dim]")
            temperature = Prompt.ask(
                "   Temperatura (0.0-1.0)",
                default="0.1"
            )
            try:
                options['temperature'] = float(temperature)
            except ValueError:
                options['temperature'] = 0.1
            console.print()

            # --- NUEVO: Captura opcional de API key ---
            console.print("[cyan]3.b Credenciales de OpenAI:[/cyan]")
            console.print("   [dim]Si la variable de entorno OPENAI_API_KEY no está configurada,[/dim]")
            console.print("   [dim]puede introducir la clave ahora (no se mostrará en pantalla).[/dim]")
            if Confirm.ask("   ¿Introducir OPENAI_API_KEY ahora?", default=False):
                api_key = getpass("   OPENAI_API_KEY: ")
                if api_key.strip():
                    options['api_key'] = api_key.strip()
            console.print()
        
        # Salida
        console.print("[cyan]4. Configuración de salida:[/cyan]")
        console.print("   [dim]Carpeta donde se guardarán los reportes[/dim]")
        output_dir = Prompt.ask(
            "   Directorio de salida",
            default="data/reports"
        )
        options['output_dir'] = output_dir
        
        console.print("   [dim]Genera HTML y PDF del informe[/dim]")
        regenerate_report = Confirm.ask(
            "   ¿Regenerar reporte HTML/PDF?",
            default=True
        )
        options['regenerate_report'] = regenerate_report
        console.print()
        
        # Confirmación
        console.print(Panel(
            "[bold yellow]⚠️ Configuración del Replay[/bold yellow]\n" +
            "\n".join([f"• {k}: {v}" for k, v in options.items()]),
            box=box.ROUNDED
        ))
        console.print()
        
        if Confirm.ask("[bold]¿Proceder con el replay?[/bold]", default=True):
            options['action'] = 'replay'
            options['case_id'] = case_id
            return options
        else:
            return {'action': 'back'}
    
    async def run_replay(self, options: Dict[str, Any]):
        """Ejecuta el replay con las opciones seleccionadas"""
        self.show_header()
        
        case_id = options['case_id']
        console.print(f"[bold cyan]🔄 Iniciando replay del caso {case_id}[/bold cyan]")
        console.print()
        
        # Barra de progreso
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # Tareas
            task1 = progress.add_task("[cyan]Cargando datos del cache...", total=100)
            await asyncio.sleep(0.5)
            progress.update(task1, completed=100)
            
            task2 = progress.add_task("[cyan]Procesando con IA...", total=100)
            
            # ✅ Ejecutar el replay REAL
            try:
                results = await self.system.replay_case(case_id, options)
                progress.update(task2, completed=100)
                
                console.print()
                console.print("[bold green]✅ Replay completado exitosamente[/bold green]")
                
                # Mostrar información real de los resultados
                if results and results.get('output_path'):
                    output_path = Path(results['output_path'])
                    
                    # Buscar archivos con el nuevo formato
                    import glob
                    html_pattern = str(output_path / f"INF-*-*.html")
                    pdf_pattern = str(output_path / f"INF-*-*.pdf")
                    
                    html_files = glob.glob(html_pattern)
                    pdf_files = glob.glob(pdf_pattern)
                    
                    # Si hay archivos con el nuevo formato, usar el más reciente
                    if html_files:
                        html_file = Path(max(html_files, key=lambda x: Path(x).stat().st_mtime))
                    else:
                        # Fallback al formato antiguo
                        html_file = output_path / f"INF-{case_id}.html"
                    
                    if pdf_files:
                        pdf_file = Path(max(pdf_files, key=lambda x: Path(x).stat().st_mtime))
                    else:
                        # Fallback al formato antiguo
                        pdf_file = output_path / f"INF-{case_id}.pdf"
                    
                    console.print(f"\n[bold]Archivos generados:[/bold]")
                    if html_file.exists():
                        console.print(f"  ✓ HTML: [green]{html_file}[/green]")
                    else:
                        console.print(f"  ✗ HTML: [red]No generado[/red]")
                        
                    if pdf_file.exists():
                        console.print(f"  ✓ PDF: [green]{pdf_file}[/green]")
                    else:
                        console.print(f"  ✗ PDF: [yellow]No generado (WeasyPrint no instalado)[/yellow]")
                
                # Mostrar métricas del procesamiento
                if results:
                    if 'extraction_results' in results:
                        console.print(f"  • Documentos procesados: {len(results['extraction_results'])}")
            
            except Exception as e:
                progress.update(task2, completed=100)
                console.print()
                console.print(f"[bold red]❌ Error durante el replay: {e}[/bold red]")
                logger.error(f"Error en replay: {e}", exc_info=True)
        
        console.print()
        Prompt.ask("Presione Enter para continuar")
    
    async def run(self):
        """Ejecuta la interfaz principal"""
        while True:
            choice = self.show_main_menu()
            
            if choice == "1":
                # Ver casos
                case_id = self.show_cases_list()
                if case_id and case_id not in ['back', 'quit']:
                    options = self.show_case_details(case_id)
                    if options.get('action') == 'replay':
                        await self.run_replay(options)
                elif case_id == 'quit':
                    break
                    
            elif choice == "2":
                # Replay directo
                case_id = Prompt.ask("[bold]Ingrese el Case ID[/bold]")
                options = self.show_case_details(case_id)
                if options.get('action') == 'replay':
                    await self.run_replay(options)
                    
            elif choice == "3":
                # Limpiar cache
                self.clean_cache_menu()
                
            elif choice == "4":
                # Estadísticas detalladas
                self.show_detailed_stats()
                
            elif choice == "5":
                # Salir
                if Confirm.ask("[yellow]¿Está seguro que desea salir?[/yellow]", default=False):
                    console.print("[cyan]👋 Hasta luego![/cyan]")
                    break
    
    def show_detailed_stats(self):
        """Muestra estadísticas detalladas del sistema"""
        self.show_header()
        
        stats = self.cache_manager.get_cache_stats()
        cases = self.cache_manager.list_cached_cases()
        
        # Calcular estadísticas adicionales
        total_docs = sum(c['total_documents'] for c in cases)
        avg_docs = total_docs / len(cases) if cases else 0
        
        # Crear layout
        console.print("[bold cyan]📊 Estadísticas Detalladas del Sistema[/bold cyan]")
        console.print()
        
        # Tabla de estadísticas
        table = Table(box=box.SIMPLE_HEAD, show_header=False)
        table.add_column("Métrica", style="cyan")
        table.add_column("Valor", style="yellow")
        
        table.add_row("Total de casos", str(stats['total_cases']))
        table.add_row("Total de archivos cacheados", str(stats['total_cached_files']))
        table.add_row("Tamaño del cache", f"{stats['cache_size_mb']} MB")
        table.add_row("Promedio docs/caso", f"{avg_docs:.1f}")
        table.add_row("Directorio de cache", stats['cache_directory'])
        
        console.print(table)
        console.print()
        
        # Top 5 casos más recientes
        if cases:
            console.print("[bold]Últimos 5 casos procesados:[/bold]")
            for case in cases[:5]:
                console.print(f"  • {case['case_id']}: {case['case_title'][:50]}")
        
        console.print()
        Prompt.ask("Presione Enter para continuar")

    # ============================
    #   NUEVO: Menú de limpieza
    # ============================
    def clean_cache_menu(self):
        """Menú para limpiar cache de casos"""
        self.show_header()
        
        cases = self.cache_manager.list_cached_cases()
        
        if not cases:
            console.print("[yellow]⚠️ No hay casos en cache[/yellow]")
            Prompt.ask("Presione Enter para continuar")
            return
        
        console.print("[bold]Casos disponibles para limpiar:[/bold]")
        for idx, case in enumerate(cases, 1):
            console.print(f"  [{idx}] {case['case_id']}: {case['case_title'][:40]}")
        
        console.print("\n  [a] Limpiar TODO el cache")
        console.print("  [v] Volver")
        
        choice = Prompt.ask("[bold yellow]Seleccione[/bold yellow]")
        
        if choice.lower() == 'v':
            return
        elif choice.lower() == 'a':
            if Confirm.ask("[red]¿Está seguro de limpiar TODO el cache?[/red]", default=False):
                try:
                    result = self.system.replay_service.clear_cache(["all"])  # Usa servicio centralizado
                    if result.get("status") == "success":
                        console.print("[green]✓ Cache limpiado completamente[/green]")
                    else:
                        console.print(f"[yellow]⚠️ {result.get('message','No se pudo limpiar completamente')}[/yellow]")
                except Exception as e:
                    console.print(f"[red]❌ Error limpiando cache: {e}[/red]")
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(cases):
                    case_id = cases[idx]['case_id']
                    if Confirm.ask(f"[yellow]¿Limpiar cache de {case_id}?[/yellow]", default=False):
                        # Implementar limpieza específica
                        try:
                            # Usar el servicio para limpiar correctamente artefactos e índice
                            result = self.system.replay_service.clear_cache([case_id])
                            if result.get("status") == "success":
                                console.print(f"[green]✓ Cache de {case_id} limpiado[/green]")
                            else:
                                console.print(f"[yellow]⚠️ {result.get('message','No se pudo limpiar completamente')}[/yellow]")
                        except Exception as e:
                            console.print(f"[red]❌ Error limpiando cache de {case_id}: {e}[/red]")
                else:
                    console.print("[red]Opción inválida[/red]")
            except ValueError:
                console.print("[red]Opción inválida[/red]")
        
        Prompt.ask("Presione Enter para continuar")
