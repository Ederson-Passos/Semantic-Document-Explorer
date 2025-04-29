import asyncio
import os
import traceback
import math

from Agents import DocumentAnalysisAgent, ReportingAgent
from Authentication import GoogleDriveAPI
from DataBaseManager import initialize_apis_and_db, list_drive_files
from dotenv import load_dotenv

from FolderManager import create_directories
from LLMManager import setup_groq_llm, initialize_llm
from ReportGeneretor import process_batches

DRIVE_FOLDER_ID = "1lXQ7R5z8NGV1YGUncVDHntiOFX35r6WO"
REPORT_DIR = "google_drive_reports"
TEMP_DIR = "temp_drive_files"
BATCH_SIZE = 2  # Define o número de arquivos por lote.


def initialize_agents(groq_chat_llm):
    """Inicializa os agentes de análise de documentos e relatórios."""
    print("Inicializando Agentes CrewAI...")
    try:
        document_agent = DocumentAnalysisAgent(llm=groq_chat_llm)
        reporting_agent = ReportingAgent(llm=groq_chat_llm)
        print("Agentes inicializados com sucesso.")
        return document_agent, reporting_agent
    except TypeError as e:
        print(f"Erro de Tipo ao inicializar agentes: {e}")
        print("Verifique se as classes DocumentAnalysisAgent e ReportingAgent em Agents.py foram atualizadas para aceitar o argumento 'llm'.")
        traceback.print_exc()
        return None, None
    except Exception as e:
        print(f"Erro inesperado ao inicializar agentes: {e}")
        traceback.print_exc()
        return None, None

async def main():
    """Função principal assíncrona que orquestra o processo de análise de documentos."""
    print("Inicializando o LLM Manager...")
    llm_manager = setup_groq_llm() # Obtém a instância do *gerenciador* GroqLLM
    if llm_manager is None:
        print("Falha ao inicializar o LLM Manager. Encerrando.")
        return

    llm_manager, groq_chat_llm = initialize_llm()
    if groq_chat_llm is None:
        return

    create_directories(REPORT_DIR, TEMP_DIR)

    drive_api = GoogleDriveAPI()
    drive_service = drive_api.service
    if not drive_service:
        print("Falha ao obter o serviço do Google Drive. Verifique a autenticação.")
        return
    db_manager = initialize_apis_and_db(drive_service)
    if db_manager is None:
        return

    document_agent, reporting_agent = initialize_agents(groq_chat_llm)
    if document_agent is None or reporting_agent is None:
        return

    files = list_drive_files(db_manager, DRIVE_FOLDER_ID)
    if files is None:
        return

    if not files:
        print("Nenhum arquivo encontrado na pasta especificada do Google Drive.")
        # Limpar diretório temporário caso algo tenha sido baixado antes
        # (Embora improvável se a listagem falhou)
        # cleanup_temp_files([], TEMP_DIR) # Função de limpeza precisa ser importada ou movida
        return
    else:
        total_files = len(files)
        # Evita divisão por zero se BATCH_SIZE for inválido
        if BATCH_SIZE <= 0:
            print("Erro: BATCH_SIZE deve ser um número positivo.")
            return
        total_batches = math.ceil(total_files / BATCH_SIZE)
        print(f"Encontrados {total_files} arquivos. Serão processados em {total_batches} lotes de até {BATCH_SIZE} "
              f"arquivos cada.")

    # Lógica de processamento em lotes.
    print("Iniciando processamento em lotes...")
    try:
        await process_batches(
            files=files,
            total_files=total_files,
            total_batches=total_batches,
            batch_size=BATCH_SIZE,
            db_manager=db_manager,
            document_agent=document_agent,
            reporting_agent=reporting_agent,
            temp_dir=TEMP_DIR,
            report_dir=REPORT_DIR,
            llm_manager=llm_manager
        )
        print("Processamento em lotes concluído.")
    except Exception as e:
        print(f"Erro durante a execução de process_batches: {e}")
        traceback.print_exc()

# --- Ponto de Entrada do Script ---
if __name__ == "__main__":
    print("Executando script principal (Main.py)...")
    print("Carregando variáveis de ambiente do arquivo .env (se existir)...")
    # Carrega variáveis do arquivo .env no diretório atual ou parent.
    # É crucial que GROQ_API_KEY esteja definida aqui ou no ambiente do sistema.
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        print("AVISO URGENTE: A variável de ambiente GROQ_API_KEY não foi encontrada!")
        print("Certifique-se de que ela está definida no seu ambiente ou em um arquivo .env.")
        exit(1)

    # Executa a função main assíncrona
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExecução interrompida pelo usuário.")
    except Exception as e:
        print(f"\nErro fatal não tratado no loop de eventos asyncio: {e}")
        traceback.print_exc()
    finally:
        print("Script principal finalizado.")