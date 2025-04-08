import os.path
import io
import os
from typing import List, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ["https://www.googleapis.com/auth/drive"]
CREDENTIALS_FILE = "credentials.json"
TOKEN_FILE = "token.json"

class GoogleDriveAPI:
    def __init__(self):
        """Inicializa a classe e tenta autenticar com o Google Drive."""
        self._processed_folders = set()
        self.service = self._authenticate()

    def _authenticate(self):
        """Autentica o usuário e retorna um objeto de serviço do Google Drive API."""
        creds = None
        # Verifica se o arquivo token.json existe e carrega as credenciais.
        if os.path.exists(TOKEN_FILE):
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

        # Se não houver credenciais válidas inicia o fluxo de autenticação.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
                creds = flow.run_local_server(port=0)
            # Salva as credenciais para futuras execuções.
            with open(TOKEN_FILE, "w") as token:
                token.write(creds.to_json())

        # Constrói e retorna o objeto de serviço do Google Drive API.
        return build("drive", "v3", credentials=creds)

    def list_files(self, folder_id: str, page_size: int = 100) -> List[dict]:
        """Lista os arquivos dentro de uma pasta específica do Google Drive.
        Args:
            folder_id: o ID da pasta do Google Drive a ser pesquisado.
            page_size: número máximo de arquivos a serem listados por página.
        Returns: Uma lista de dicionários, onde cada dicionário representa um arquivo."""
        try:
            query = f"'{folder_id}' in parents and mimeType != 'application/vnd.google-apps.folder'"

            results = self.service.files().list(q=query, pageSize=page_size, fields="nextPageToken, files(id, name)").execute()
            items = results.get("files", [])

            if not items:
                print(f"Nenhum arquivo encontrado na pasta com ID: {folder_id}")
                return []

            print(f"Arquivos na pasta (ID: {folder_id}):")
            for item in items:
                print(f"- {item['name']} ({item['id']})")
            return items
        except HttpError as e:
            print(f"Ocorreu um erro ao listar os arquivos da pasta {folder_id}: {e}")
            return []

    def list_files_recursively(self, folder_id: str) -> List[dict]:
        """Lista todos os arquivos dentro de uma pasta específica e de todas as subpastas."""
        # Limpa o set de pastas processadas no início de uma nova chamada de alto nível.
        if not hasattr(self, '_processed_folders') or folder_id not in self._processed_folders:
            if hasattr(self, '_processed_folders'):
                self._processed_folders.clear()
            else:
                self._processed_folders = set()

        all_files = []
        page_token = None

        # Adiciona a pasta atual ao set de processadas para evitar ciclos repetitivos
        self._processed_folders.add(folder_id)
        print(f"--- Buscando na pasta: {folder_id} ---")

        # Loop para lidar com a paginação dentro da pasta atual
        while True:
            try:
                # Query para buscar arquivos e pastas dentro da pasta atual
                query = f"'{folder_id}' in parents"
                results = self.service.files().list(
                    q=query,
                    pageSize=100,
                    fields="nextPageToken, files(id, name, mimeType)",
                    pageToken=page_token
                ).execute()

                items = results.get("files", [])
                print(f"Itens encontrados nesta página da pasta {folder_id}: {len(items)}")

                for item in items:
                    item_id = item.get('id')
                    item_name = item.get('name', 'Nome Desconhecido')

                    # Se for uma subpasta, será acessada (recursão)
                    if 'mimetype' in item and item['mimetype'] == 'application/vnd.google-apps.folder':
                        # Verifica se já foi processada anteriormente
                        if item_id not in self._processed_folders:
                            print(f"Recursão -> Entrando na subpasta: '{item_name}' (ID: {item_id})")
                            # Chamada recursiva
                            sub_folder_files = self.list_files_recursively(item_id)
                            all_files.extend(sub_folder_files) # Adiciona os arquivos encontrados na subpasta
                        else:
                            print(f"Aviso: Pasta '{item_name}' (ID: {item_id}) já processada, pulando para evitar loop.")

                    # É um arquivo ao invés de uma subpasta, adiciona à lista
                    # Ou não possui mimetype
                    else:
                        print(f"Encontrado item (não pasta): '{item_name}' (ID: {item_id}), Mimetype: {item.get('mimetype')}")
                        all_files.append({'id': item_id, 'name': item_name})

                page_token = results.get('nextPageToken', None)
                if page_token is None:
                    break # Não há mais páginas, sai do laço while

            except HttpError as e:
                print(f"!! Erro de API ao listar itens na pasta {folder_id}: {e}")
                print("!! Continuando a busca onde possível...")
                # Pára de processar esta pasta e suas subpastas
                break
            except Exception as e:
                print(f"!! Erro inesperado ao processar pasta {folder_id}: {e}")
                break # Sai do loop while

        print(f"--- Finalizando busca na pasta: {folder_id} ---")
        return all_files


    def download_file(self, file_id: str, file_name: str, destination_path=".") -> bool:
        """Baixa um arquivo específico do Google Drive."""
        try:
            # Garante que o diretório de destino exista
            os.makedirs(destination_path, exist_ok=True)

            request = self.service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            print(f"Iniciando download de '{file_name}'...")
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    # Atualiza o progresso na mesma linha
                    print(f"\r Download {int(status.progress() * 100)}%...", end='')
            print(f"\r Download 100% concluído.") # Limpa a linha e mostra 100%
            file_path = os.path.join(destination_path, file_name)
            with open(file_path, "wb") as f:
                f.write(fh.getvalue())
            print(f"Arquivo '{file_name}' (ID: {file_id}) baixado para '{file_path}'.")
            return True
        except HttpError as error:
            print(f"\nOcorreu um erro ao baixar o arquivo (ID: {file_id}): {error}")
            return False
        except Exception as e:
            print(f"\n Erro inesperado durante o download (ID: {file_id}): {e}")
            return False

    def download_multiple_files(self, file_ids_and_names: List[str], destination_path="."):
        """Baixa múltiplos arquivos do Google Drive."""
        # Garante que o diretório de destino exista
        os.makedirs(destination_path, exist_ok=True)
        total_files = len(file_ids_and_names)
        print(f"\nIniciando download de {total_files} arquivo(s) para '{destination_path}'...")
        downloaded_count = 0
        error_count = 0

        for i, file_info in enumerate(file_ids_and_names):
            file_id = file_info.get('id')
            file_name = file_info.get('name')

            if not file_id or not file_name:
                print(f"Informação inválida para o arquivo {i + 1}/{total_files}. Pulando.")
                error_count += 1
                continue

            print(f"\nProcessando arquivo {i + 1}/{total_files}: '{file_name}' (ID: {file_id})")

            if self.download_file((file_id, file_name, destination_path)):
                downloaded_count += 1
            else:
                error_count += 1

        print(f"\n--- Resumo do Download Múltiplo ---")
        print(f"Total de arquivos para baixar: {total_files}")
        print(f"Baixados com sucesso: {downloaded_count}")
        print(f"Erros ou pulados: {error_count}")
        print(f"-----------------------------------")

if __name__ == "__main__":
    TARGET_FOLDER_ID = "1YJFQnLQfz30ZAXoR8CWSlKGkkEupcPfm"

    drive_api = GoogleDriveAPI()

    # Listar arquivos RECURSIVAMENTE a partir da pasta alvo
    print(f"\n=== Iniciando Listagem Recursiva a partir da Pasta ID: {TARGET_FOLDER_ID} ===")
    # Chama o NOVO método recursivo
    all_files_recursive = drive_api.list_files_recursively(folder_id=TARGET_FOLDER_ID)
    print(f"\n=== Listagem Recursiva Concluída ===")
    print(f"Total de arquivos encontrados em '{TARGET_FOLDER_ID}' e subpastas: {len(all_files_recursive)}")

    if all_files_recursive:
        # Mostrar os primeiros 2 arquivos encontrados como exemplo
        print("\nPrimeiros 2 arquivos encontrados:")
        for f in all_files_recursive[:2]:
            print(f" - {f.get('name', 'Nome Indisponível')} (ID: {f.get('id', 'ID Indisponível')})")

    else:
        print(f"\nNenhum arquivo foi encontrado recursivamente dentro da pasta {TARGET_FOLDER_ID}.")
