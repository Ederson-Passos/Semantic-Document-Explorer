import os.path
import io
import os
import time
import traceback
from io import BytesIO
from typing import List, Optional

from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

from TextExtractor import TEMP_DOWNLOAD_FOLDER

# Diretório onde os arquivos serão baixados temporariamente
DOWNLOAD_FOLDER = TEMP_DOWNLOAD_FOLDER

GOOGLE_DOCS_EXPORT_MIMETYPES = {
    'application/vnd.google-apps.document': 'application/pdf',
    'application/vnd.google-apps.spreadsheet': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'application/vnd.google-apps.presentation': 'application/pdf'
}

EXPORT_EXTENSIONS = {
    'application/pdf': '.pdf',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx'
}

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

    def _get_file_metadata(self, file_id:  str) -> Optional[dict]:
        """Busca metadados básicos de um arquivo, incluindo mimeType."""
        try:
            file_metadata = self.service.files().get(
                fileId=file_id, fields='id, name, mimeType'
            ).execute()
            return file_metadata
        except HttpError as e:
            print(f"Erro ao buscar metadados para file ID {file_id}: {e}")
            return None
        except Exception as e:
            print(f"Erro inesperado ao buscar metadados para file ID {file_id}: {e}")
            return None

    def _prepare_file_download(self, file_id: str, file_name: str) -> Optional[tuple[str, str, str]]:
        """
        Prepara o download de um arquivo, determinando o tipo de exportação necessário e o nome final do arquivo.
        Args:
            file_id: ID do arquivo no Google Drive.
            file_name: Nome original do arquivo.
        Returns:
            Uma tupla contendo (original_mime_type, export_mime_type, final_file_name) ou None em caso de erro.
        """
        print(f"Preparando para baixar/exportar '{file_name}' (ID: {file_id})...")
        metadata = self._get_file_metadata(file_id)

        if not metadata:
            print(f"Não foi possível obter metadados para '{file_name}', download cancelado.")
            return None

        original_mime_type = metadata.get('mimeType')
        export_mime_type = GOOGLE_DOCS_EXPORT_MIMETYPES.get(original_mime_type)

        # Ajusta o nome do arquivo e define o caminho final.
        base_name, _ = os.path.splitext(file_name)
        if export_mime_type:
            # Adiciona um fallback caso o export_mime_type não seja encontrado.
            extension = EXPORT_EXTENSIONS.get(export_mime_type)
            final_file_name = f"{base_name}{extension}"
            print(f"Arquivo '{file_name}' é um Google Doc ({original_mime_type}). Exportando como {export_mime_type} "
                  f"para '{final_file_name}'...")
        else:
            final_file_name = file_name
            print(f"Arquivo '{file_name}' ({original_mime_type}) será baixado diretamente.")

        return original_mime_type, export_mime_type, final_file_name

    def _download_or_export_file(
            self,
            file_id: str,
            export_mime_type: Optional[str],
            final_file_name: str
        ) -> tuple[None, None, None, None] | tuple[BytesIO, bool, MediaIoBaseDownload, int]:
        """
        Realiza o download ou exportação do arquivo, dependendo do tipo de MIME.
        Args:
            file_id: ID do arquivo no Google Drive.
            export_mime_type: Tipo MIME para exportação (se aplicável).
            final_file_name: Nome final do arquivo.
        Returns:
            Uma tupla contendo o objeto BytesIO com o conteúdo do arquivo e um booleano
            indicando se o download foi concluído.
        """
        try:
            if export_mime_type:
                # Lógica de exportação.
                request = self.service.files().export_media(
                    fileId=file_id, mimeType=export_mime_type
                )
            else:
                # Lógica de download direto.
                request = self.service.files().get_media(fileId=file_id)

            # Processo de download/exportação.
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            chunk_num = 0
            print(f"Iniciando download de '{final_file_name}'...")
            return fh, done, downloader, chunk_num

        except HttpError as error:
            print(f"\nOcorreu um erro de API ao iniciar o download/exportação do arquivo (ID: {file_id}, "
                  f"Nome: {final_file_name}): {error}")
            # Log extra para erro 403:
            if error.resp.status == 403:
                print("Detalhes do erro 403:", error.content)
            return None, None, None, None
        except Exception as e:
            print(f"\n Erro inesperado durante o download (ID: {file_id}): {e}")
            traceback.print_exc()
            return None, None, None, None

    def download_file(self, file_id: str, file_name: str, destination_path=".") -> Optional[str]:
        """Baixa ou exporta um arquivo específico do Google Drive para um diretório local."""
        print(f"Preparando para baixar/exportar '{file_name}' (ID: {file_id})...")
        metadata = self._get_file_metadata(file_id)

        if not metadata:
            print(f"Não foi possivel obter metadados para '{file_name}', download cancelado.")
            return None

        # Prepara o download do arquivo.
        preparation_result = self._prepare_file_download(file_id, file_name)
        if not preparation_result:
            return None
        original_mime_type, export_mime_type, final_file_name = preparation_result

        # Garante que o diretório de destino exista.
        if not _ensure_destination_directory(destination_path):
            return None

        file_path = os.path.join(destination_path, final_file_name)
        fh, done, downloader, chunk_num = self._download_or_export_file(file_id, export_mime_type, final_file_name)

        if fh is None:
            return None

        if not _download_chunks(downloader, final_file_name, fh):
            return None

        _save_file(file_path, fh)
        print(f"\r [Download '{file_name}'] Loop de chunks concluído. Salvando arquivo...")
        print(f"Arquivo '{file_name}' (ID: {file_id}) baixado para '{file_path}'.")
        return file_path

def _ensure_destination_directory(destination_path: str) -> bool:
    """
    Garante que o diretório de destino exista.

    Args:
        destination_path: Caminho do diretório de destino.

    Returns:
        True se o diretório existe ou foi criado com sucesso, False caso contrário.
    """
    try:
        os.makedirs(destination_path, exist_ok=True)
        return True
    except OSError as e:
        print(f"Erro ao criar/verificar diretório de destino '{destination_path}': {e}")
        return False

def _download_chunks(downloader: MediaIoBaseDownload, file_name: str, fh: BytesIO) -> bool:
    """
    Baixa o arquivo em chunks e lida com erros de forma robusta.
    Args:
        downloader: Objeto MediaIoBaseDownload para gerenciar o download.
        file_name: Nome do arquivo sendo baixado.
        fh: Objeto BytesIO para armazenar o conteúdo do arquivo.
    Returns:
        True se o download foi bem-sucedido, False caso contrário.
    """
    done = False
    chunk_num = 0
    while not done:
        chunk_num += 1
        status = None
        try:
            print(f"\r [Download/Export '{file_name}'] Solicitando chunk "
                  f"{chunk_num}...", end='', flush=True)
            status, done = downloader.next_chunk(num_retries=3)
            print(f"\r [Download/Export '{file_name}'] Recebido. Status={type(status)}, Done={done}. "
                  f"Processando...", end='', flush=True)
            if status:
                print(f"\r [Download/Export '{file_name}'] Progresso: {int(status.progress() * 100)}% "
                      f"(Chunk {chunk_num} recebido)...", end="", flush=True)
            else:
                print(f"\r [Download/Export '{file_name}'] Chunk {chunk_num} recebido (status None). "
                      f"Done={done}", end='', flush=True)

        except HttpError as chunk_error:
            # Verifica se o erro é recuperável antes de falhar.
            if chunk_error.resp.status >= 500:
                print(f"\n [Download '{file_name}'] Erro HTTP recuperável no chunk {chunk_num}: "
                      f"{chunk_error}. Tentando novamente...")
                time.sleep(min(2**chunk_num, 30))  # Backoff exponencial simples. Limita a 30 s.
            else:
                print(f"\n [Download '{file_name}'] Erro HTTP não recuperável "
                      f"no chunk {chunk_num}: {chunk_error}")
                return False
        except Exception as chunk_generic_error:
            print(f"\n [Download '{file_name}'] Erro genérico no chunk {chunk_num}: {chunk_generic_error}")
            traceback.print_exc()
            return False
    return True

def _save_file(file_path: str, fh: BytesIO):
    """Salva o conteúdo do BytesIO em um arquivo."""
    with open(file_path, "wb") as f:
        f.write(fh.getvalue())

def initialize_apis_and_db(drive_service):
    """Inicializa as APIs e o gerenciador de banco de dados."""
    print("Inicializando APIs e Gerenciador de Banco de Dados...")
    try:
        db_manager = DataBaseManager(drive_service)
        return db_manager
    except Exception as e:
        print(f"Erro durante a inicialização da API do Google Drive ou DataBaseManager: {e}")
        traceback.print_exc()
        return None

def list_drive_files(db_manager, drive_folder_id):
    """Lista os arquivos da pasta especificada no Google Drive."""
    print(f"Listando arquivos da pasta Google Drive ID: {drive_folder_id}")
    try:
        if hasattr(db_manager, '_processed_folders'):
            db_manager._processed_folders.clear()
        files = db_manager.list_files_recursively(drive_folder_id)
        return files
    except Exception as e:
        print(f"Erro ao listar arquivos do Google Drive: {e}")
        traceback.print_exc()
        return None