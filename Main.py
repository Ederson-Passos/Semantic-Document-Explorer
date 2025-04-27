import asyncio
import os
import traceback
import math

from Agents import DocumentAnalysisAgent, ReportingAgent
from Authentication import GoogleDriveAPI
from DataBaseManager import DataBaseManager
from dotenv import load_dotenv
from LLMManager import setup_groq_llm
from ReportGeneretor import process_batches

DRIVE_FOLDER_ID = "1lXQ7R5z8NGV1YGUncVDHntiOFX35r6WO"
REPORT_DIR = "google_drive_reports"
TEMP_DIR = "temp_drive_files"
BATCH_SIZE = 2  # Define o número de arquivos por lote.

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

async def main():
    llm = setup_groq_llm()
    if llm is None:
        return

    # Inicialização de APIs e Agentes
    drive_api = GoogleDriveAPI()
    drive_service = drive_api.service
    db_manager = DataBaseManager(drive_service)

    try:
        document_agent = DocumentAnalysisAgent(llm=llm)
        reporting_agent = ReportingAgent(llm=llm)
    except TypeError:
        print("Erro: Verifique se a classe DocumentAnalysisAgent e ReportingAgent estão configuradas corretamente.")
        traceback.print_exc()
        return
    except Exception as e:
        print(f"Erro ao inicializar agentes: {e}")
        traceback.print_exc()
        return

    # Obtém a lista de arquivos do Google Drive.
    print(f"Listando arquivos da pasta Google Drive ID: {DRIVE_FOLDER_ID}")
    try:
        files = db_manager.list_files_recursively(DRIVE_FOLDER_ID)
    except Exception as e:
        print(f"Erro ao listar arquivos do Google Drive: {e}")
        traceback.print_exc()
        return

    if not files:
        print("Nenhum arquivo encontrado na pasta especificada do Google Drive.")
        return
    else:
        total_files = len(files)
        total_batches = math.ceil(total_files / BATCH_SIZE)
        print(f"Encontrados {total_files} arquivos. Serão processados em {total_batches} lotes de até {BATCH_SIZE} "
              f"arquivos cada.")

    # Lógica de processamento em lotes.
    await process_batches(files, total_files, total_batches, BATCH_SIZE, db_manager, document_agent, reporting_agent,
                          TEMP_DIR, REPORT_DIR)


if __name__ == "__main__":
    print("Executando script principal...")
    print("Certifique-se de que a variável de ambiente GROQ_API_KEY está definida.")
    print("Considere usar um arquivo .env e a biblioteca python-dotenv para gerenciá-la.")
    load_dotenv() # Carrega variáveis do arquivo .env no diretório atual
    asyncio.run(main())
    print("Script principal finalizado.")