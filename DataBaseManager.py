import os.path
import io
import os
from typing import List

from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

from TextExtractor import process_and_tokenize_file, TEMP_DOWNLOAD_FOLDER
from EmbeddingGenerator import EmbeddingGenerator

# Diretório onde os arquivos serão baixados temporariamente
DOWNLOAD_FOLDER = TEMP_DOWNLOAD_FOLDER

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

    def process_and_embed_all_files_recursively(self, folder_id: str) -> dict:
        """
        Lista todos os arquivos recursivamente, baixa, extrai o texto, tokeniza e gera embeddings para
        o conteúdo de cada arquivo usando TensorFlow.
        Args:
            folder_id (str): O ID da pasta a ser processada.
        Returns:
            dict: Um dicionário onde as chaves são os nomes dos arquivos e os valores são os caminhos para
            os arquivos de embedding correspondentes.
        """
        all_files = self.list_files_recursively(folder_id)
        embedding_paths = {}
        embedding_generator = EmbeddingGenerator()

        for file_info in all_files:
            file_id = file_info.get('id')
            file_name = file_info.get('name')

            if not file_id or not file_name:
                print(f"Informação de arquivo inválida: {file_info}")
                continue

            download_path = os.path.join(DOWNLOAD_FOLDER, file_name)
            if self.download_file(file_id, file_name, DOWNLOAD_FOLDER):
                _, tokens = process_and_tokenize_file(download_path)
                if tokens:
                    # Processa os tokens em partes gerenciáveis (chunks de 510 tokens, deixando
                    # espaço para [CLS] e [SEP]) e gera embeddings para cada parte.
                    chunk_size = 510
                    num_chunks = (len(tokens) + chunk_size - 1) // chunk_size # Descarta o resto da divisão
                    all_chunk_embeddings_paths = []

                    for i in range(num_chunks):
                        start_index = i * chunk_size
                        end_index = min((i + 1) * chunk_size, len(tokens))
                        chunk_tokens = tokens[start_index:end_index]
                        if chunk_tokens:
                            embedding_filename = f"{os.path.splitext(file_name)[0]}_part_{i}"
                            embedding_path = embedding_generator.generate_embeddings(chunk_tokens, embedding_filename)
                            all_chunk_embeddings_paths.append(embedding_path)

                    embedding_paths[file_name] = all_chunk_embeddings_paths
                os.remove(download_path) # Limpa o arquivo baixado após o processamento
        return embedding_paths

    def cleanup_temp_folder(self):
        """Limpa o diretório temporário de download."""
        for filename in os.listdir(DOWNLOAD_FOLDER):
            file_path = os.path.join(DOWNLOAD_FOLDER, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Erro ao remover {file_path}: {e}")
        os.rmdir(DOWNLOAD_FOLDER)
        print(f"Diretório temporário '{DOWNLOAD_FOLDER}' limpo.")
