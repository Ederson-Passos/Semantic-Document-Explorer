import os.path
import io
from typing import List

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

    def list_files(self, page_size = 10) -> List[dict]:
        """Lista os arquivos do Google Drive.
        Args: page_size: número máximo de arquivos a serem listados por página.
        Returns: Uma lista de dicionários, onde cada dicionário representa um arquivo."""
        try:
            results = self.service.files().list(pageSize=page_size, fields="nextPageToken, files(id, name)").execute()
            items = results.get("files", [])

            if not items:
                print("Nenhum arquivo encontrado.")
                return []
            print("Arquivos:")
            for item in items:
                print(f"- {item['name']} ({item['id']}")
            return items
        except HttpError as e:
            print(f"Ocorreu um erro ao listar os arquivos: {e}")
            return []

    def download_file(self, file_id: str, file_name: str, destination_path=".") -> bool:
        """Baixa um arquivo específico do Google Drive."""
        try:
            request = self.service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                print(f"Download {int(status.progress() * 100)}%.")
            file_path = os.path.join(destination_path, file_name)
            with open(file_path, "wb") as f:
                f.write(fh.getvalue())
            print(f"Arquivo '{file_name}' (ID: {file_id}) baixado para '{file_path}'.")
            return True
        except HttpError as error:
            print(f"Ocorreu um erro ao baixar o arquivo (ID: {file_id}): {error}")
            return False

    def download_multiple_files(self, file_ids: List[str], destination_path="."):
        """Baixa múltiplos arquivos do Google Drive."""
        for file_id in file_ids:
            try:
                file = self.service.files().get(fileId=file_id, fields="name").execute()
                file_name = file.get("name")
                if file_name:
                    self.download_file(file_id, file_name, destination_path)
                else:
                    print(f"Não foi possível obter o nome do arquivo com ID: {file_id}")
            except HttpError as error:
                print(f"Ocorreu um erro ao obter informações do arquivo (ID: {file_id}): {error}")

if __name__ == "__main__":
    drive_api = GoogleDriveAPI()

    # Listar arquivos
    files = drive_api.list_files(page_size=50)
    if files:
        # Exemplo de IDs de arquivos para download
        file_ids_to_download = [file['id'] for file in files[:2]] # Baixar os dois primeiros arquivos listados

        # Baixar um arquivo específico (você precisará do ID do arquivo)
        if files:
            first_file_id = files[0]['id']
            first_file_name = files[0]['name']
            drive_api.download_file(first_file_id, f"baixado_{first_file_name}")

        # Baixar múltiplos arquivos
        drive_api.download_multiple_files(file_ids_to_download, "arquivos_baixados")
