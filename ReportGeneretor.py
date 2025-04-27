"""
Contém a lógica para gerar relatórios.
"""
import datetime
import os
import traceback

from crewai import Crew, Process, Task
from crewai.tools import BaseTool
from FolderManager import cleanup_temp_files


class GenerateReportTool(BaseTool):
    name: str = "generate_report"
    description: str = "Generates a report summarizing the key findings from the document analysis."

    def _run(self, analysis_results: dict, report_directory: str = "reports") -> str:
        """
        Generates a report based on the analysis results.
        Args:
            analysis_results (dict): a dictionary containing the results of the document analysis.
            report_directory (str): the dictionary where the report should be saved.
        Returns:
            str: the path to the generated report.
        """

        os.makedirs(report_directory, exist_ok=True)
        report_path = os.path.join(report_directory, "report.txt")
        with open(report_path, "w") as f:
            f.write("Document Analysis Report\n")
            f.write("-----------------------\n\n")
            for file_path, results in analysis_results.items():
                f.write(f"File: {os.path.basename(file_path)}\n")
                f.write(f"  Word Count: {results.get('word_count', 'N/A')}\n")
                f.write(f"  Summary: {results.get('summary', 'N/A')}\n")
                f.write("\n")

                # Adicionar um relatório mais sofisticado (visualização de dados, análise de tendência)

        return report_path


async def process_batches(files, total_files, total_batches, batch_size, db_manager, document_agent,
                          reporting_agent, temp_dir, report_dir):
    """Processa os arquivos em lotes, criando tarefas de análise e relatórios parciais."""
    all_partial_reports = []
    all_downloaded_files = []

    for i in range(0, total_files, batch_size):
        current_batch_files = files[i:i + batch_size]
        batch_number = (i // batch_size) + 1
        print(f"\nIniciando lote {batch_number} de {total_batches} ({len(current_batch_files)} arquivos)...")

        batch_tasks = []
        batch_downloaded_files = []

        print(f" Baixando arquivos e criando tarefas de análise para o lote {batch_number}...")
        for file in current_batch_files:
            process_file_in_batch(file, batch_number, db_manager, document_agent, batch_tasks, batch_downloaded_files, all_downloaded_files, temp_dir)

        if not any(task.agent == document_agent for task in batch_tasks):
            print(f"      Nenhuma tarefa de análise criada para o lote {batch_number}. Pulando para o próximo lote.")
            cleanup_temp_files(batch_downloaded_files, temp_dir)  # Limpar por lote.
            continue  # Pula para a próxima iteração do loop de lotes

        # Criar tarefa de relatório parcial para o lote.
        partial_report_task = create_partial_report_task(batch_number, batch_tasks, reporting_agent)
        batch_tasks.append(partial_report_task)  # Adiciona a tarefa de relatório parcial às tarefas do lote.

        # Executar Crew por lote.
        await run_batch_crew(batch_number, batch_tasks, document_agent, reporting_agent, all_partial_reports, batch_downloaded_files, temp_dir)

    consolidate_and_save_reports(all_partial_reports, report_dir)
    cleanup_temp_files(all_downloaded_files, temp_dir)


def create_partial_report_task(batch_number, batch_tasks, reporting_agent):
    """
    Cria uma tarefa de relatório parcial para um lote específico.
    """
    print(f"  Criando tarefa de relatório parcial para o lote {batch_number}...")
    partial_report_task = Task(
        description=(
            f"Consolide os resultados das {len(batch_tasks)} análises de documentos deste lote"
            f" (Lote {batch_number}). Crie um relatório parcial que liste "
            "o resumo e a contagem de palavras para cada arquivo deste lote."
        ),
        expected_output=(
            f"Um relatório de texto único (.txt) que resume a análise (resumo e contagem) de cada documento "
            f"processado no Lote {batch_number}."
        ),
        agent=reporting_agent,  # Usa o agente de relatório modificado
        context=batch_tasks  # O contexto são apenas as tarefas deste lote.
    )
    return partial_report_task


def process_file_in_batch(file, batch_number, db_manager, document_agent, batch_tasks, batch_downloaded_files,
                          all_downloaded_files, temp_dir):
    """
    Processa um único arquivo dentro de um lote, baixando-o e criando uma tarefa de análise.
    """
    safe_filename = (
        file["name"].replace("/", "_").replace("\\", "_").replace(":", "_").replace("*", "_").replace("?", "_")
        .replace("\"", "_").replace("<", "_").replace(">", "_").replace("|", "_")
    )
    file_path = os.path.join(temp_dir, safe_filename)
    print(f"    Processando arquivo: {safe_filename} (ID: {file['id']})")
    try:
        if db_manager.download_file(file["id"], safe_filename, temp_dir):
            batch_downloaded_files.append(file_path)
            all_downloaded_files.append(file_path)  # Adiciona à lista geral também

            if os.path.exists(file_path):
                print(f"      Arquivo '{safe_filename}' baixado para '{file_path}'.")
                analysis_task = Task(
                    description=(
                        f"Analise o conteúdo do documento '{safe_filename}' (Lote {batch_number}). "
                        f"O arquivo está localizado em: '{file_path}'. "
                        "Sua análise deve incluir: 1. Contagem total de palavras. 2. Um resumo conciso (2-3 frases)."
                    ),
                    expected_output=(
                        f"Um relatório de análise conciso para '{safe_filename}' (Lote {batch_number}) contendo:\n"
                        "- A contagem total de palavras.\n"
                        "- Um resumo do conteúdo (aproximadamente 2-3 frases)."
                    ),
                    agent=document_agent,
                )
                batch_tasks.append(analysis_task)
            else:
                print(f"      Falha ao verificar a existência do arquivo baixado: {file_path}")
        else:
            print(f"      Falha ao baixar o arquivo: {file['name']}")
    except Exception as e:
        print(f"      Erro ao processar o arquivo {file['name']} no lote {batch_number}: {e}")
        traceback.print_exc()


async def run_batch_crew(batch_number, batch_tasks, document_agent, reporting_agent, all_partial_reports,
                         batch_downloaded_files, temp_dir):
    """Executes the Crew for a specific batch and handles the results."""
    print(f"  Criando e executando Crew para o lote {batch_number} com {len(batch_tasks)} tarefas...")
    batch_crew = Crew(
        agents=[document_agent, reporting_agent],
        tasks=batch_tasks,
        process=Process.sequential,
        verbose=True,
        memory=False
    )

    try:
        # O resultado do kickoff do lote é o relatório parcial.
        partial_report = batch_crew.kickoff()
        print(f"  Execução do Crew para o lote {batch_number} concluída.")

        # Coletar relatórios parciais.
        if partial_report:
            print(f"    Relatório parcial do lote {batch_number} gerado.")
            all_partial_reports.append(f"--- Relatório do Lote {batch_number} ---\n{str(partial_report)}")
        else:
            print(f"    Atenção: Relatório parcial do lote {batch_number} veio vazio ou nulo.")
            all_partial_reports.append(f"--- Relatório do Lote {batch_number} (FALHA OU VAZIO) ---")

    except Exception as e:
        print(f"\n  Erro durante a execução do Crew (kickoff) para o lote {batch_number}: {e}")
        traceback.print_exc()
        all_partial_reports.append(f"--- Relatório do Lote {batch_number} (ERRO NA EXECUÇÃO: {e}) ---")

    finally:
        cleanup_temp_files(batch_downloaded_files, temp_dir)


def consolidate_and_save_reports(all_partial_reports, report_dir):
    """
    Consolida todos os relatórios parciais em um único relatório final e o salva em um arquivo.

    Args:
        all_partial_reports (list): Uma lista de relatórios parciais (strings).
        report_dir (str): O diretório onde o relatório final será salvo.
    """
    print("\n--- Consolidação Final dos Relatórios ---")
    if all_partial_reports:
        print(f"Combinando {len(all_partial_reports)} relatórios parciais...")
        # Combina os relatórios parciais com um separador claro.
        final_report_content = "\n\n========================================\n\n".join(all_partial_reports)

        # Salva o relatório final combinado.
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(report_dir, f"FINAL_consolidated_drive_report_{timestamp}.txt")
        try:
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(final_report_content)
            print(f"Relatório final consolidado salvo em: {report_file}")
        except Exception as e:
            print(f"Erro ao salvar o relatório final consolidado em {report_file}: {e}")
            traceback.print_exc()
