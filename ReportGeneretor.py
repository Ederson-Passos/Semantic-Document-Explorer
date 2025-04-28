"""
Contém a lógica para gerar relatórios por lotes de arquivos.
"""
import asyncio
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


def process_batch_results(results_from_gather, batch_number, all_downloaded_files_overall, temp_dir):
    """
    Processes the results from asyncio.gather for a batch of files.
    """
    batch_tasks = []
    batch_downloaded_files = []
    print(f"   Processamento paralelo do lote {batch_number} concluído. Coletando resultados...")
    for result in results_from_gather:
        if isinstance(result, BaseException):
            print(f"   Erro durante o processamento paralelo de um arquivo no lote {batch_number}: {result}")
            if isinstance(result, Exception):
                traceback.print_exc()
            continue

        if result and len(result) == 2:
            analysis_task, downloaded_file_path = result
            if analysis_task and downloaded_file_path:
                batch_tasks.append(analysis_task)
                batch_downloaded_files.append(downloaded_file_path)
                all_downloaded_files_overall.append(downloaded_file_path)
            else:
                print(f"   Processamento de arquivo no lote {batch_number} não retornou tarefa/caminho válidos "
                      f"(recebido: {result!r}).")
        elif result is not None:
            print(f"   Resultado inesperado recebido do processamento no lote {batch_number}: {result!r}")

    if not batch_tasks:
        print(f"      Nenhuma tarefa de análise criada com sucesso para o lote {batch_number}. Pulando para o "
              f"próximo lote.")
        cleanup_temp_files(batch_downloaded_files, temp_dir)
    return batch_tasks, batch_downloaded_files

async def process_batches(files, total_files, total_batches, batch_size, db_manager, document_agent,
                          reporting_agent, temp_dir, report_dir):
    """
    Processa os arquivos em lotes, criando tarefas de análise e relatórios parciais.
    O processamento de arquivos dentro de cada lote é paralelizado.
    """
    all_partial_reports = []
    all_downloaded_files_overall = []

    for i in range(0, total_files, batch_size):
        current_batch_files = files[i:i + batch_size]
        batch_number = (i // batch_size) + 1
        print(f"\nIniciando lote {batch_number} de {total_batches} ({len(current_batch_files)} arquivos)...")

        # Execução com paralelização.
        batch_tasks_coroutines = []  # Lista para guardar as corrotinas de processamento de arquivo.
        print(f" Iniciando processamento paralelo para o lote {batch_number}...")
        for file in current_batch_files:
            # Cria a corrotina para processar cada arquivo e adiciona à lista.
            coro = process_file_in_batch(
                file,
                batch_number,
                db_manager,
                document_agent,
                temp_dir
            )
            batch_tasks_coroutines.append(coro)

        # Executa todas as corrotinas do lote concorrentemente.
        # Lista de tuplas: [(task, file_path), (None, None), ...]
        results_from_gather = await asyncio.gather(*batch_tasks_coroutines, return_exceptions=True)

        # Processa os resultados após a conclusão do gather.
        batch_tasks, batch_downloaded_files = process_batch_results(
            results_from_gather, batch_number, all_downloaded_files_overall, temp_dir
        )
        if not batch_tasks:
            continue
        # Cria tarefa de relatório parcial para o lote.
        # Passa as tarefas que foram realmente criadas.
        partial_report_task = create_partial_report_task(batch_number, batch_tasks, reporting_agent)
        # Adiciona a tarefa de relatório parcial às tarefas do lote.
        tasks_for_crew = batch_tasks + [partial_report_task]

        # Executar Crew por lote.
        await run_batch_crew(batch_number, tasks_for_crew, document_agent, reporting_agent, all_partial_reports,
                             batch_downloaded_files, temp_dir)

    consolidate_and_save_reports(all_partial_reports, report_dir)
    cleanup_temp_files(all_downloaded_files_overall, temp_dir)


def create_partial_report_task(batch_number, analysis_tasks, reporting_agent):
    """
    Cria uma tarefa de relatório parcial para um lote específico.
    """
    print(f"   Criando tarefa de relatório parcial para o lote {batch_number}...")
    # Garante que analysis_tasks seja uma lista, mesmo que vazia.
    analysis_tasks = analysis_tasks or []
    partial_report_task = Task(
        description=(
            f"Consolide os resultados das {len(analysis_tasks)} análises de documentos deste lote"
            f" (Lote {batch_number}). Crie um relatório parcial que liste "
            "o resumo e a contagem de palavras para cada arquivo deste lote."
        ),
        expected_output=(
            f"Um relatório de texto único (.txt) que resume a análise (resumo e contagem) de cada documento "
            f"processado no Lote {batch_number}."
        ),
        agent=reporting_agent,
        context=analysis_tasks  # O contexto são apenas as tarefas deste lote.
    )
    return partial_report_task


async def process_file_in_batch(file, batch_number, db_manager, document_agent, temp_dir) -> tuple | None:
    """
    Processa um único arquivo: baixa (em thread separada) e cria uma tarefa de análise.
    Retorna uma tupla (analysis_task, file_path) em caso de sucesso, ou None em caso de falha.
    """
    safe_filename = (
        file["name"].replace("/", "_").replace("\\", "_").replace(":", "_").replace("*", "_").replace("?", "_")
        .replace("\"", "_").replace("<", "_").replace(">", "_").replace("|", "_")
    )
    file_path = os.path.join(temp_dir, safe_filename)
    print(f"    [Lote {batch_number}] Iniciando processamento para: {safe_filename} (ID: {file['id']})")

    # download_success = False
    try:
        # Executa a função de download síncrona em um thread separado para não bloquear o loop de eventos.
        print(f"      [Lote {batch_number}] Agendando download para: {safe_filename}")
        download_success = await asyncio.to_thread(
            db_manager.download_file, file["id"], safe_filename, temp_dir
        )

        if download_success:
            # Verifica se o arquivo realmente existe após o download.
            if await asyncio.to_thread(os.path.exists, file_path):
                print(f"      [Lote {batch_number}] Download concluído e verificado: {safe_filename}")
                # Cria a tarefa de análise.
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
                # Retorna a tarefa criada e o caminho do arquivo.
                return analysis_task, file_path
            else:
                print(f"      [Lote {batch_number}] Falha ao verificar a existência do arquivo após "
                      f"download: {file_path}")
                return None, None  # Falha na verificação.
        else:
            print(f"      [Lote {batch_number}] Falha no download (retorno da função foi False): {safe_filename}")
            return None, None  # Falha no download.

    except Exception as e:
        print(f"      [Lote {batch_number}] Erro EXCEPCIONAL ao processar o arquivo {safe_filename}: {e}")
        traceback.print_exc()
        # Retorna None para indicar falha, mesmo que o download tenha ocorrido antes da exceção.
        # A limpeza será feita com base nos arquivos encontrados no diretório temp.
        return None, None


async def run_batch_crew(batch_number, batch_tasks, document_agent, reporting_agent, all_partial_reports,
                         batch_downloaded_files, temp_dir):
    """Executes the Crew for a specific batch and handles the results."""
    # Garante que batch_tasks seja uma lista.
    batch_tasks = batch_tasks or []
    if not batch_tasks:
        print(f"  Nenhuma tarefa válida para executar no Crew do lote {batch_number}.")
        # Limpa os arquivos baixados para este lote, pois não haverá relatório
        cleanup_temp_files(batch_downloaded_files, temp_dir)
        return # Não executa o Crew se não houver tarefas

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
        partial_report = await asyncio.to_thread(batch_crew.kickoff)  # Executa o kickoff síncrono em thread.
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
        print(f"  Iniciando limpeza pós-Crew para o lote {batch_number}...")
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
