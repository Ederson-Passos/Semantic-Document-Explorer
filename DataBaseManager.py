import os.path
import io
import os
from typing import List

from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

from TextExtractor import process_and_tokenize_file, TEMP_DOWNLOAD_FOLDER

# Diretório onde os arquivos serão baixados temporariamente
DOWNLOAD_FOLDER = TEMP_DOWNLOAD_FOLDER

def cleanup_temp_folder():
    """Limpa o diretório temporário de download."""
    for filename in os.listdir(DOWNLOAD_FOLDER):
        file_path = os.path.join(DOWNLOAD_FOLDER, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Erro ao remover {file_path}: {e}")
    if os.path.exists(DOWNLOAD_FOLDER):
        os.rmdir(DOWNLOAD_FOLDER)
        print(f"Diretório temporário '{DOWNLOAD_FOLDER}' limpo.")


class DataBaseManager:
    """
    Classe para interagir com a API do Google Drive v3, fornecendo serviços como
    listar, baixar e processar arquivos.
    """
    def __init__(self, service):
        """
        Inicializa a classe com o objeto de serviço autenticado do Google Drive API.
        Args:
            service: Objeto de serviço autenticado do Google Drive API v3.
        """
        self.service = service
        self._processed_folders = set()
        os.makedirs(DOWNLOAD_FOLDER, exist_ok=True) # Garante que o diretório temporário exista

    def list_files(self, folder_id: str, page_size: int = 100) -> List[dict]:
        """Lista os arquivos (não pastas) dentro de uma pasta específica do Google Drive."""
        if not self.service:
            print("Erro: Serviço do Google Drive não inicializado.")
            return []
        try:
            query = f"'{folder_id}' in parents and mimeType != 'application/vnd.google-apps.folder'"
            results = self.service.files().list(
                q=query,
                pageSize=page_size,
                fields="nextPageToken, files(id, name)"
            ).execute()
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
        except Exception as e:
            print(f"Ocorreu um erro inesperado ao listar arquivos: {e}")
            return []

    def _scan_items(self, items: List[dict], all_files: List[dict]):
        """
        Varre os itens retornados pela API do Google Drive.
        Processa pastas recursivamente e adiciona arquivos a uma lista.
        Args:
            items: Lista de itens retornados pela API do Google Drive.
            all_files: Lista para armazenar os arquivos encontrados.
        """
        for item in items:
            item_id = item.get('id')
            item_name = item.get('name', 'Nome Desconhecido')
            mime_type = item.get('mimeType')

            if mime_type == 'application/vnd.google-apps.folder':
                if item_id not in self._processed_folders:
                    print(f"Recursão -> Entrando na subpasta: '{item_name}' (ID: {item_id})")
                    sub_folder_files = self.list_files_recursively(item_id)
                    all_files.extend(sub_folder_files)
                else:
                    print(f"Aviso: Pasta '{item_name}' (ID: {item_id}) já processada, pulando.")
            else:
                print(f"Encontrado arquivo: '{item_name}' (ID: {item_id}), Mimetype: {mime_type}")
                all_files.append({'id': item_id, 'name': item_name})

    def list_files_recursively(self, folder_id: str) -> List[dict]:
        """Lista todos os arquivos dentro de uma pasta e subpastas de forma recursiva."""
        if not self.service:
            print("Erro: Serviço do Google Drive não inicializado.")
            return []

        if not hasattr(self, '_processed_folders') or folder_id not in self._processed_folders:
            if hasattr(self, '_processed_folders'):
                self._processed_folders.clear()
            else:
                self._processed_folders = set()

        all_files = []
        page_token = None

        self._processed_folders.add(folder_id)
        print(f"--- Buscando na pasta: {folder_id} ---")

        while True:
            try:
                query = f"'{folder_id}' in parents"
                results = self.service.files().list(
                    q=query,
                    pageSize=100,
                    fields="nextPageToken, files(id, name, mimeType)",
                    pageToken=page_token
                ).execute()

                items = results.get("files", [])
                print(f"Itens encontrados nesta página da pasta {folder_id}: {len(items)}")

                # Percorrer todos os itens encontrados.
                self._scan_items(items, all_files)

                page_token = results.get('nextPageToken', None)
                if page_token is None:
                    break

            except HttpError as e:
                print(f"Erro de API ao listar itens na pasta {folder_id}: {e}")
                print("Continuando a busca onde possível...")
                break
            except Exception as e:
                print(f"Erro inesperado ao processar pasta {folder_id}: {e}")
                break

        print(f"--- Finalizando busca na pasta: {folder_id} ---")
        return all_files

    def download_file(self, file_id: str, file_name: str, destination_path=".") -> bool:
        """Baixa um arquivo específico do Google Drive para um diretório local."""
        try:
            os.makedirs(destination_path, exist_ok=True)
            file_path = os.path.join(destination_path, file_name)
            request = self.service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            print(f"Iniciando download de '{file_name}'...")
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    print(f"\r Download {int(status.progress() * 100)}%...", end='')
            print(f"\r Download 100% concluído.")
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
