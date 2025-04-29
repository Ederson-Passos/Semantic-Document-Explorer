"""
Contém a lógica para gerar relatórios por lotes de arquivos.
"""
import asyncio
import datetime
import os
import traceback
from typing import Dict, Any, Optional, List

from crewai import Crew, Process, Task, Agent
from crewai.tools import BaseTool

from DataBaseManager import DataBaseManager
from FolderManager import cleanup_temp_files


class GenerateReportTool(BaseTool):
    name: str = "save_analysis_report"
    description: str = ("Saves the provided analysis results dictionary to a text file. "
                        "Input should be a dictionary where keys are filenames and values are analysis results.")

    def _run(self, analysis_results: dict, report_directory: str = "reports", filename_prefix: str = "analysis") -> str:
        """
        Saves the analysis results dictionary to a file.
        Args:
            analysis_results (dict): Dictionary containing analysis results.
            report_directory (str): Directory to save the report.
            filename_prefix (str): Prefix for the report filename.
        Returns:
            str: Path to the saved report file.
        """
        if not isinstance(analysis_results, dict):
            return "Error: Input must be a dictionary."

        try:
            # Cria o diretório especificado pela variável 'report_directory'.
            # O argumento 'exist_ok=True' garante que nenhum erro será lançado se o diretório já existir.
            os.makedirs(report_directory, exist_ok=True)
            # Obtém a data e hora atuais e formata como uma string no formato "AnoMesDia_HoraMinutoSegundo".
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"{filename_prefix}_report_{timestamp}.txt"  # Cria o nome do arquivo de relatório.
            report_path = os.path.join(report_directory, report_filename)  # Constrói o caminho.

            with open(report_path, "w", encoding="utf-8") as f:
                f.write(f"Document Analysis Report - {timestamp}\n")
                f.write("========================================\n\n")
                for file_key, results in analysis_results.items():
                    # Assume que file_key pode ser o nome do arquivo ou um identificador
                    f.write(f"File: {file_key}\n")
                    # Verifica se 'results' é um dicionário antes de usar .get()
                    if isinstance(results, dict):
                        f.write(f"  Word Count: {results.get('word_count', 'N/A')}\n")
                        f.write(f"  Summary: {results.get('summary', 'N/A')}\n")
                    else:
                        # Se o resultado não for um dicionário (ex: string de erro da CrewAI)
                        f.write(f"  Analysis Result: {results}\n")
                    f.write("-" * 40 + "\n\n")
            print(f"Report saved to: {report_path}")
            return report_path
        except Exception as e:
            error_message = f"Error saving report: {e}"
            print(error_message)
            traceback.print_exc()
            return error_message

async def process_file_in_batch(
        file: Dict[str, str],
        batch_number: int,
        db_manager: DataBaseManager,
        document_agent: Agent,
        temp_dir:str
) -> Optional[tuple[Task, str]]:
    """
    Processa um único arquivo: baixa (em thread separada) e cria uma tarefa de análise.
    Retorna uma tupla (analysis_task, file_path) em caso de sucesso, ou None em caso de falha.
    """
    file_id = file.get('id')
    original_filename = file.get('name')

    if not file_id or not original_filename:
        print(f"    [Lote {batch_number}] Erro: Informações inválidas para o arquivo: {file}. Pulando.")
        return None

    # Trata o nome do arquivo para evitar problemas com caracteres inválidos no path.
    safe_filename = "".join(c if c.isalnum() or c in ('_', '.', '-') else '_' for c in original_filename)
    # Garante que não comece com '.' ou '_' se o original não começava, para evitar arquivos ocultos inesperados.
    if original_filename and not original_filename.startswith(('.', '_')) and safe_filename.startswith(('.', '_')):
        safe_filename = "file_" + safe_filename.lstrip('._')

    file_path = os.path.join(temp_dir, safe_filename)
    print(f"    [Lote {batch_number}] Iniciando processamento para: {safe_filename} "
          f"(Original: {original_filename}, ID: {file_id})")

    download_success = False
    try:
        # Executa download síncrono em thread
        print(f"      [Lote {batch_number}] Agendando download para: {safe_filename}")
        download_success = await asyncio.to_thread(
            db_manager.download_file, file_id, safe_filename, temp_dir # Passa safe_filename para download.
        )
        print(f"      [Lote {batch_number}] Tentativa de download de {safe_filename} finalizada "
              f"(Sucesso reportado: {download_success}).")

        if download_success:
            # Verifica se o arquivo realmente existe após a tentativa de download.
            file_exists = await asyncio.to_thread(os.path.exists, file_path)

            if file_exists:
                print(f"      [Lote {batch_number}] Download concluído e arquivo verificado: {safe_filename}")
                # Cria a tarefa de análise CrewAI.
                analysis_task = Task(
                    description=(
                        f"Analise o conteúdo do documento '{safe_filename}' (do Lote {batch_number}). "
                        f"O arquivo está localizado em: '{file_path}'. "
                        "Sua análise deve incluir: 1. Contagem total de palavras. 2. Um resumo conciso (aproximadamente 2-3 frases)."
                    ),
                    expected_output=(
                        f"Um dicionário Python contendo a análise para '{safe_filename}' (Lote {batch_number}) com as chaves:\n"
                        "- 'word_count': (int) A contagem total de palavras.\n"
                        "- 'summary': (str) Um resumo do conteúdo (aproximadamente 2-3 frases)."
                        # O output esperado deve ser o que o LLM retorna, não o nome do arquivo final.
                    ),
                    agent=document_agent,
                )
                print(f"      [Lote {batch_number}] Tarefa de análise para {safe_filename} criada.")
                # Retorna a tarefa criada e o caminho do arquivo.
                return analysis_task, file_path
            else:
                # Se download_success foi True, mas o arquivo não existe, algo deu errado.
                print(f"      [Lote {batch_number}] ERRO INESPERADO: Download reportado como sucesso, "
                      f"mas arquivo não encontrado em: {file_path}")
                return None # Falha na verificação pós-download.
        else:
            # Se download_success foi False.
            print(f"      [Lote {batch_number}] Falha no download (retorno da função foi False): {safe_filename}")
            return None # Falha no download.

    except Exception as e:
        print(f"      [Lote {batch_number}] Erro EXCEPCIONAL durante processamento do arquivo {safe_filename}: {e}")
        traceback.print_exc()
        # Retorna None para indicar falha. A limpeza ocorrerá no final do lote.
        return None

def process_batch_results(results_from_gather: list, batch_number: int):
    """
    Processa os resultados brutos do asyncio.gather para um lote.
    Separa tarefas bem-sucedidas, arquivos baixados e lida com erros.
    Return:
        tuple[List[Task], List[str]]: Uma tupla contendo a lista de tarefas de análise válidas
                                      e a lista de caminhos de arquivos baixados com sucesso.
    """
    batch_analysis_tasks = []
    batch_downloaded_files = []
    print(f"   [Lote {batch_number}] Processamento paralelo concluído. Coletando e validando resultados...")

    for result in results_from_gather:
        if isinstance(result, BaseException):
            # Se o gather retornou uma exceção diretamente.
            print(f"   [Lote {batch_number}] Erro capturado pelo asyncio.gather: {result!r}")
            if isinstance(result, Exception):  # Loga traceback para Exceptions "reais".
                traceback.print_exc()
            continue  # Pula para o próximo resultado.

        # Verifica se o resultado é a tupla esperada (Task, str).
        if result and isinstance(result, tuple) and len(result) == 2:
            analysis_task, downloaded_file_path = result
            # Verifica se ambos os elementos da tupla são válidos.
            if (analysis_task and isinstance(analysis_task, Task) and downloaded_file_path and
                    isinstance(downloaded_file_path, str)):
                batch_analysis_tasks.append(analysis_task)
                batch_downloaded_files.append(downloaded_file_path)
                print(f"     - Resultado válido coletado para: {os.path.basename(downloaded_file_path)}")
            else:
                # Caso onde process_file_in_batch retornou (None, None) ou algo inválido na tupla.
                print(f"   [Lote {batch_number}] Processamento de arquivo não retornou tarefa/caminho válidos "
                      f"(recebido: {result!r}).")
        elif result is not None:
            # Log para tipos de resultado inesperados que não são Exceptions nem tuplas válidas.
            print(f"   [Lote {batch_number}] Resultado inesperado recebido do processamento paralelo: {result!r}")
        # else: result is None  # Já logado dentro de process_file_in_batch como falha

    print(f"   [Lote {batch_number}] Coleta finalizada. {len(batch_analysis_tasks)} tarefas válidas, "
          f"{len(batch_downloaded_files)} arquivos baixados.")
    return batch_analysis_tasks, batch_downloaded_files

def create_report_task(batch_number: int, analysis_tasks: List[Task], reporting_agent: Any):
    """
    Cria a tarefa final de relatório para um lote específico, usando as tarefas de análise como contexto.
    """
    if not analysis_tasks:
        print(f"   [Lote {batch_number}] Nenhuma tarefa de análise para incluir no relatório.")
        return None # Não cria tarefa de relatório se não houver análises.

    print(f"   [Lote {batch_number}] Criando tarefa de relatório com {len(analysis_tasks)} análises como contexto...")
    report_task = Task(
        description=(
            f"Consolide os resultados das {len(analysis_tasks)} tarefas de análise de documentos fornecidas como contexto "
            f"(representando o Lote {batch_number}). Para cada documento analisado, extraia as informações relevantes "
            "(como nome do arquivo implícito na descrição da tarefa de análise, contagem de palavras e resumo) "
            "do resultado da respectiva tarefa de análise. "
            "Formate a saída como um relatório único e coeso em formato Markdown, com uma seção para cada documento."
        ),
        expected_output=(
            "Um relatório em formato Markdown. O relatório deve ter:\n"
            "- Um título geral (ex: 'Relatório de Análise - Lote {batch_number}').\n"
            "- Uma seção para CADA documento analisado, contendo:\n"
            "  - O nome do arquivo (tente inferir da descrição da tarefa de análise).\n"
            "  - A contagem de palavras encontrada.\n"
            "  - O resumo gerado.\n"
            "- Use formatação Markdown clara (ex: cabeçalhos `###`, listas, etc.)."
        ),
        agent=reporting_agent,
        context=analysis_tasks # Passa as tarefas de análise concluídas como contexto para esta tarefa.
    )
    return report_task

async def run_batch_crew(
        batch_number: int,
        analysis_tasks: List[Task],
        report_task: Optional[Task],
        document_agent: Any,
        reporting_agent: Any
) -> Optional[str]:
    """
    Executa a CrewAI para um lote específico, incluindo tarefas de análise e a tarefa de relatório.
    O kickoff síncrono é executado em um thread separado.
    Return:
        Optional[str]: O conteúdo do relatório gerado pela tarefa final (report_task), ou None se falhar.
    """
    if not analysis_tasks and not report_task:
        print(f"  [Lote {batch_number}] Nenhuma tarefa válida para executar no Crew.")
        return None # Não executa o Crew se não houver tarefas

    tasks_for_crew = analysis_tasks + ([report_task] if report_task else [])
    print(f"  [Lote {batch_number}] Criando Crew com {len(tasks_for_crew)} tarefas ({len(analysis_tasks)} análise(s), "
          f"{'1 relatório' if report_task else 'sem relatório'})...")

    # Definindo os agentes que participarão.
    agents_for_crew = [document_agent, reporting_agent]

    batch_crew = Crew(
        agents=agents_for_crew,
        tasks=tasks_for_crew,
        process=Process.sequential, # Executa tarefas em ordem: primeiro análises, depois relatório
        verbose=True,
        memory=False
    )

    final_report_content = None
    try:
        print(f"  [Lote {batch_number}] Executando Crew.kickoff() em thread separada...")
        # O resultado da crew será o resultado da última tarefa na lista (report_task, se existir)
        kickoff_result = await asyncio.to_thread(batch_crew.kickoff)
        print(f"  [Lote {batch_number}] Execução do Crew.kickoff() concluída.")

        # Verifica o resultado. Se report_task existia, espera-se que kickoff_result seja o conteúdo do relatório.
        if report_task:
            if kickoff_result and isinstance(kickoff_result, str):
                final_report_content = kickoff_result
                print(f"    [Lote {batch_number}] Relatório gerado com sucesso.")
            elif kickoff_result:
                print(f"    [Lote {batch_number}] Atenção: Resultado do kickoff não foi uma string como esperado "
                      f"para o relatório: {type(kickoff_result)}")
                final_report_content = (f"--- Relatório do Lote {batch_number} (FORMATO INESPERADO) "
                                        f"---\n{str(kickoff_result)}")
            else:
                print(f"    [Lote {batch_number}] Atenção: Relatório do lote veio vazio ou nulo após kickoff.")
                final_report_content = f"--- Relatório do Lote {batch_number} (FALHA OU VAZIO) ---"
        else:
            print(f"    [Lote {batch_number}] Crew executada apenas para análise (sem tarefa de relatório). "
                  f"Resultado do kickoff: {kickoff_result}")


    except Exception as e:
        print(f"\n  [Lote {batch_number}] ERRO CRÍTICO durante a execução do Crew.kickoff(): {e}")
        traceback.print_exc()
        # Registra o erro para o relatório final
        final_report_content = f"--- Relatório do Lote {batch_number} (ERRO NA EXECUÇÃO DA CREW: {e}) ---"

    return final_report_content

def save_final_report(all_reports_content: List[str], report_dir: str):
    """
    Consolida todos os conteúdos de relatórios (strings) em um único arquivo final.
    """
    print("\n--- Consolidação e Salvamento do Relatório Final ---")
    if not all_reports_content:
        print("Nenhum conteúdo de relatório parcial foi gerado para consolidar.")
        return

    print(f"Combinando {len(all_reports_content)} seções de relatório...")
    # Combina os conteúdos com um separador claro.
    final_report_text = "\n\n========================================\n\n".join(all_reports_content)

    # Salva o relatório final combinado.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"FINAL_consolidated_drive_report_{timestamp}.md" # Salva como Markdown.
    report_filepath = os.path.join(report_dir, report_filename)
    try:
        with open(report_filepath, "w", encoding="utf-8") as f:
            f.write(f"# Relatório Consolidado de Análise de Documentos - {timestamp}\n\n")
            f.write(final_report_text)
        print(f"Relatório final consolidado salvo em: {report_filepath}")
    except IOError as e:
        print(f"Erro ao salvar o relatório final consolidado em {report_filepath}: {e}")
        traceback.print_exc()
    except Exception as e:
        print(f"Erro inesperado ao salvar o relatório final: {e}")
        traceback.print_exc()

# Função principal de orquestração de lotes.
async def process_batches(
        files: List[Dict[str, str]],
        total_files: int,
        total_batches: int,
        batch_size: int,
        db_manager: DataBaseManager,
        document_agent: Any, # Tipagem pode ser Agent
        reporting_agent: Any, # Tipagem pode ser Agent
        temp_dir: str,
        report_dir: str
):
    """
    Orquestra o processamento dos arquivos em lotes:
    1. Para cada lote, processa arquivos em paralelo (download + criação de task de análise).
    2. Coleta os resultados do processamento paralelo.
    3. Cria uma tarefa de relatório para o lote (se houver análises).
    4. Executa a CrewAI para o lote (análises + relatório).
    5. Coleta o conteúdo do relatório gerado.
    6. Limpa os arquivos temporários do lote.
    7. Após todos os lotes, consolida e salva o relatório final.
    """
    all_final_reports_content = []  # Lista para guardar o conteúdo (string) de cada relatório de lote.
    all_downloaded_files_ever = set()  # Usar set para evitar duplicatas e facilitar limpeza final

    for i in range(0, total_files, batch_size):
        start_index = i
        end_index = min(i + batch_size, total_files)
        current_batch_files = files[start_index:end_index]
        batch_number = (i // batch_size) + 1
        print(f"\n--- Iniciando Lote {batch_number} de {total_batches} ({len(current_batch_files)} arquivos) ---")

        # 1. Processamento paralelo dos arquivos do lote
        batch_coroutines = []
        print(f"  [Lote {batch_number}] Agendando processamento paralelo de arquivos...")
        for file_info in current_batch_files:
            coro = process_file_in_batch(
                file_info, batch_number, db_manager, document_agent, temp_dir
            )
            batch_coroutines.append(coro)

        # Executa as corotinas de process_file_in_batch concorrentemente
        gather_results = await asyncio.gather(*batch_coroutines, return_exceptions=True)

        # 2. Coleta e validação dos resultados
        batch_analysis_tasks, batch_downloaded_files = process_batch_results(gather_results, batch_number)
        # Adiciona os arquivos baixados neste lote ao conjunto geral
        all_downloaded_files_ever.update(batch_downloaded_files)

        # 3. Criação da tarefa de relatório (se houver tarefas de análise)
        batch_report_task = create_report_task(batch_number, batch_analysis_tasks, reporting_agent)

        # 4. Execução da CrewAI para o lote
        # Passa as tarefas de análise e a tarefa de relatório (que pode ser None)
        report_content = await run_batch_crew(
            batch_number,
            batch_analysis_tasks,
            batch_report_task,
            document_agent,
            reporting_agent
        )

        # 5. Coleta do conteúdo do relatório
        if report_content:
            all_final_reports_content.append(report_content)
        elif batch_analysis_tasks: # Se houve análise mas o relatório falhou/veio vazio
            all_final_reports_content.append(f"--- Relatório do Lote {batch_number} (CONTEÚDO NÃO GERADO OU INVÁLIDO) ---")
        # Se não houve nem análise, não adiciona nada

        # 6. Limpeza dos arquivos temporários DESTE lote
        print(f"  [Lote {batch_number}] Iniciando limpeza de arquivos temporários do lote...")
        # Passa a lista específica deste lote para limpeza
        cleanup_temp_files(batch_downloaded_files, temp_dir)
        print(f"--- Lote {batch_number} Finalizado ---")

    # 7. Consolidação e salvamento do relatório final após todos os lotes
    save_final_report(all_final_reports_content, report_dir)

    # Limpeza final opcional (embora a limpeza por lote deva ter pego tudo)
    print("\n--- Limpeza Final ---")
    # Converte o set para lista para a função cleanup
    cleanup_temp_files(list(all_downloaded_files_ever), temp_dir)
    print("Processamento de todos os lotes concluído.")