import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from DataBaseManager import DataBaseManager

# Define os escopos de permissão necessários para acessar o Google Drive.
SCOPES = ["https://www.googleapis.com/auth/drive"]
# Nome do arquivo que contém as credenciais da API do Google Cloud (OAuth 2.0 client ID).
CREDENTIALS_FILE = "credentials.json"
# Nome do arquivo onde o token de acesso do usuário será armazenado após a autenticação.
TOKEN_FILE = "token.json"

class GoogleDriveAPI:
    """
    Classe para autenticar e obter o serviço da API do Google Drive v3.
    """
    def __init__(self):
        """Inicializa a classe e inicia o processo de autenticação."""
        self.service = self._authenticate()
        # self.drive_service = DataBaseManager(self.service) # Cria uma instância do gerenciamento

    def _authenticate(self):
        """
        Autentica o usuário e retorna um objeto de serviço do Google Drive API.
        Retorna um objeto "resource" do Google API Client Library para interagir com o Drive.
        """
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

if __name__ == "__main__":
    drive_api = GoogleDriveAPI()
    print("Serviço do Google Drive autenticado com sucesso (dentro de Authentication.py).")