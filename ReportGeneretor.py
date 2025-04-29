"""
Contém a lógica para gerar relatórios por lotes de arquivos.
"""
import asyncio
import datetime
import os
import traceback
import threading
import fitz
from typing import Dict, Any, Optional, List
from crewai import Crew, Process, Task, Agent
from DataBaseManager import DataBaseManager
from FolderManager import cleanup_temp_files
from LLMManager import GroqLLM

def blocking_download_wrapper(
        db_manager: DataBaseManager,
        file_id: str,
        original_filename: str,
        temp_dir: str,
        batch_number: int
) -> Optional[str]:
    """
    Wrapper para a função de download que será executada em uma thread separada.
    """
    thread_id = threading.get_ident()  # Obtém ID do thread atual
    print(f"        [Thread:{thread_id} | Lote:{batch_number}] Iniciando db_manager.download_file "
          f"para {original_filename}")
    try:
        # Chama a função de download original
        result_path = db_manager.download_file(file_id, original_filename, temp_dir)
        if result_path:
            print(f"        [Thread:{thread_id} | Lote:{batch_number}] db_manager.download_file para "
                  f"{original_filename} retornou sucesso: {result_path}")
        else:
            print(f"        [Thread:{thread_id} | Lote:{batch_number}] db_manager.download_file para "
                  f"{original_filename} retornou falha (None).")
        return result_path  # Retorna o caminho ou None.
    except Exception as thread_e:
        # Captura e loga qualquer erro DENTRO da execução do thread
        print(f"        [Thread:{thread_id} | Lote:{batch_number}] ERRO DENTRO da thread de download "
              f"para {original_filename}: {thread_e}")
        traceback.print_exc()
        return None  # Indica falha

async def download_file_with_timeout(
        db_manager: DataBaseManager,
        file_id: str,
        original_filename: str,
        temp_dir: str,
        batch_number: int,
        timeout_seconds: int = 180
) -> Optional[str]:
    """
    Baixa um arquivo com timeout, usando uma thread separada.
    """
    print(f"      [Lote {batch_number}] Preparando para executar download em thread: {original_filename}")
    download_wrapper_args = (db_manager, file_id, original_filename, temp_dir, batch_number)
    download_task_wrapper = asyncio.to_thread(blocking_download_wrapper, *download_wrapper_args)
    print(f"      [Lote {batch_number}] Aguardando download de {original_filename} (max {timeout_seconds}s)...")
    try:
        actual_downloaded_path = await asyncio.wait_for(download_task_wrapper, timeout=timeout_seconds)
        return actual_downloaded_path
    except asyncio.TimeoutError:
        print(f"      [Lote {batch_number}] TIMEOUT! Download de {original_filename} excedeu {timeout_seconds} "
              f"segundos.")
        return None

async def verify_file_existence(actual_downloaded_path: str, batch_number: int) -> bool:
    """
    Verifica se o arquivo baixado existe, usando uma thread separada.
    """
    actual_filename = os.path.basename(actual_downloaded_path)
    print(f"      [Lote {batch_number}] Verificando existência do arquivo {actual_filename} em thread...")
    file_exists = await asyncio.to_thread(os.path.exists, actual_downloaded_path)
    print(f"      [Lote {batch_number}] Verificação de existência para {actual_filename} concluída: {file_exists}")
    return file_exists

def extract_and_truncate_content(
        actual_downloaded_path: str,
        actual_filename: str,
        batch_number: int,
        llm_manager: GroqLLM,
        max_input_tokens: int = 3000
) -> tuple[str, int]:
    """
    Extrai o conteúdo do arquivo, conta os tokens e trunca se necessário.
    """
    file_content = ""
    extraction_error = None
    token_count = 0
    try:
        if actual_downloaded_path.lower().endswith(".pdf"):
            try:
                with fitz.open(actual_downloaded_path) as doc:
                    file_content = "".join(page.get_text() for page in doc)
                print(f"      [Lote {batch_number}] Extraído texto do PDF: {actual_filename}")
            except Exception as pdf_error:
                extraction_error = f"Erro ao extrair texto do PDF: {pdf_error}"
                print(f"      [Lote {batch_number}] ERRO ao extrair texto do PDF {actual_filename}: {pdf_error}")
        else:  # Assume outros formatos como texto.
            try:
                with open(actual_downloaded_path, "r", encoding="utf-8", errors='ignore') as f:
                    file_content = f.read()
                print(f"      [Lote {batch_number}] Lido conteúdo (como texto) de: {actual_filename}")
            except Exception as read_error:
                extraction_error = f"Erro ao ler arquivo como texto: {read_error}"
                print(f"      [Lote {batch_number}] ERRO ao ler {actual_filename} como texto: {read_error}")

        if extraction_error:
            file_content = f"ERRO DURANTE A EXTRAÇÃO DE CONTEÚDO: {extraction_error}"

        token_count = llm_manager.contador_de_tokens(file_content)
        print(f"      [Lote {batch_number}] Contagem de tokens: {token_count} para {actual_filename}")
        file_content = llm_manager.truncate_content(file_content, token_count, max_input_tokens, batch_number)

    except Exception as e:
        print(f"      [Lote {batch_number}] ERRO GERAL durante extração/contagem de tokens para {actual_filename}: {e}")
        traceback.print_exc()
        file_content = f"ERRO CRÍTICO AO PROCESSAR CONTEÚDO DO ARQUIVO: {e}"
    return file_content, token_count

async def process_file_in_batch(
        file: Dict[str, str],
        batch_number: int,
        db_manager: DataBaseManager,
        document_agent: Agent,
        temp_dir:str,
        llm_manager: GroqLLM
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
    print(f"    [Lote {batch_number}] Iniciando processamento para: {original_filename} (ID: {file_id})")
    # Variável para guardar o caminho retornado.
    actual_downloaded_path: Optional[str] = None
    downloaded_task_wrapper = None  # Para referência em caso de timeout.
    try:
        actual_downloaded_path = await download_file_with_timeout(
            db_manager, file_id, original_filename, temp_dir, batch_number
        )
        # Lógica Pós-Download
        if actual_downloaded_path:
            # O nome do arquivo real agora é o basename do caminho retornado.
            actual_filename = os.path.basename(actual_downloaded_path)
            print(f"      [Lote {batch_number}] Download reportado como sucesso. Caminho: {actual_downloaded_path}")

            file_exists = await verify_file_existence(actual_downloaded_path, batch_number)

            if file_exists:
                print(f"      [Lote {batch_number}] Arquivo verificado: {actual_filename}. Extraindo conteúdo...")
                # Ler conteúdo, contar tokens e truncar.
                MAX_INPUT_TOKENS = 3000
                file_content = ""
                extraction_error = None
                file_content, token_count = extract_and_truncate_content(
                    actual_downloaded_path, actual_filename, batch_number, llm_manager
                )
                # Cria a tarefa de análise CrewAI
                analysis_task = Task(
                    description=(
                        f"Analise o seguinte conteúdo extraído do documento '{actual_filename}' "
                        f"(originalmente '{original_filename}', Lote {batch_number}). "
                        f"O conteúdo PODE TER SIDO TRUNCADO devido ao tamanho ({token_count} tokens fornecidos). "
                        "Sua análise DEVE focar em:\n"
                        "1. Calcular a contagem total de palavras do texto fornecido.\n"
                        "2. Gerar um resumo EXTREMAMENTE CONCISO do conteúdo principal em 5 a 10 linhas de texto.\n"
                        "NÃO inclua o texto original na sua resposta final. Retorne APENAS o dicionário Python.\n"
                        f"\nCONTEÚDO PARA ANÁLISE:\n---\n{file_content}\n---"
                    ),
                    expected_output=(
                        f"UM ÚNICO DICIONÁRIO Python contendo a análise para '{actual_filename}' (Lote {batch_number}) "
                        "com EXATAMENTE as seguintes chaves:\n"
                        "- 'word_count': (int) A contagem total de palavras do texto fornecido.\n"
                        "- 'summary': (str) O resumo conciso do conteúdo (5 a 10 linhas).\n"
                        "Não inclua nenhuma outra informação ou texto fora deste dicionário."
                    ),
                    agent=document_agent,
                )
                print(f"      [Lote {batch_number}] Tarefa de análise para {actual_filename} criada.")
                return analysis_task, actual_downloaded_path
            else:
                print(f"      [Lote {batch_number}] ERRO INESPERADO: Download reportado como sucesso, "
                      f"mas arquivo não encontrado em: {actual_downloaded_path}")
                return None
        else:
            # Se actual_downloaded_path for None.
            print(f"      [Lote {batch_number}] Download falhou ou não concluído para: {original_filename}")
            return None

    except Exception as e:
        # Captura outros erros que podem ocorrer na própria coroutine process_file_in_batch
        print(f"      [Lote {batch_number}] Erro EXCEPCIONAL durante processamento "
              f"(fora da thread de download) do arquivo {original_filename}: {e}")
        traceback.print_exc()
        return None

def _handle_gather_exception(result: BaseException, batch_number: int):
    """
    Handles exceptions returned directly by asyncio.gather.
    """
    print(f"   [Lote {batch_number}] Erro capturado pelo asyncio.gather: {result!r}")
    if isinstance(result, Exception):  # Loga traceback para Exceptions "reais".
        traceback.print_exc()

def _validate_and_collect_result(result: tuple, batch_number: int, batch_analysis_tasks: list, batch_downloaded_files: list):
    """
    Validates and collects a single result from asyncio.gather.
    """
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

def _handle_unexpected_result(result: Any, batch_number: int):
    """
    Handles unexpected result types from asyncio.gather.
    """
    print(f"   [Lote {batch_number}] Resultado inesperado recebido do processamento paralelo: {result!r}")


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
            _handle_gather_exception(result, batch_number)
            continue
        if result and isinstance(result, tuple) and len(result) == 2:
            _validate_and_collect_result(result, batch_number, batch_analysis_tasks, batch_downloaded_files)
        elif result is not None:
            _handle_unexpected_result(result, batch_number)

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
        document_agent: Any,
        reporting_agent: Any,
        temp_dir: str,
        report_dir: str,
        llm_manager: GroqLLM
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
                file_info, batch_number, db_manager, document_agent, temp_dir, llm_manager
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
            all_final_reports_content.append(f"--- Relatório do Lote {batch_number} "
                                             f"(CONTEÚDO NÃO GERADO OU INVÁLIDO) ---")

        # 6. Limpeza dos arquivos temporários DESTE lote
        print(f"  [Lote {batch_number}] Iniciando limpeza de arquivos temporários do lote...")
        # Passa a lista específica deste lote para limpeza
        cleanup_temp_files(batch_downloaded_files, temp_dir)
        print(f"--- Lote {batch_number} Finalizado ---")

    # 7. Consolidação e salvamento do relatório final após todos os lotes
    save_final_report(all_final_reports_content, report_dir)

    print("\n--- Limpeza Final ---")
    cleanup_temp_files(list(all_downloaded_files_ever), temp_dir)
    print("Processamento de todos os lotes concluído.")