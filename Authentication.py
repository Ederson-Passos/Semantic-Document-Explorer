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
        self.drive_service = DataBaseManager(self.service) # Cria uma instância do gerenciamento

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
    # Substituir pelo id da pasta alvo.

    TARGET_FOLDER_ID = "1h4ZbeAtdo5uYEqYHkzdz_0kBSt2DW0A9"

    # Instancia a classe GoogleDriveAPI, disparando a autenticação e criando o serviço.
    drive_api = GoogleDriveAPI()

    # Utiliza a classe de serviço para realizar as operações do Google Drive.
    print(f"\n=== Iniciando Listagem Recursiva a partir da Pasta ID: {TARGET_FOLDER_ID} ===")
    all_files_recursive = drive_api.drive_service.list_files_recursively(folder_id=TARGET_FOLDER_ID)
    print(f"\n=== Listagem Recursiva Concluída ===")
    print(f"Total de arquivos encontrados em '{TARGET_FOLDER_ID}' e subpastas: {len(all_files_recursive)}")

    if all_files_recursive:
        print("\nLista de todos os arquivos encontrados:")
        for f in all_files_recursive:
            print(f" - {f.get('name', 'Nome Indisponível')} (ID: {f.get('id', 'ID Indisponível')})")
    else:
        print(f"\nNenhum arquivo foi encontrado recursivamente dentro da pasta {TARGET_FOLDER_ID}.")

    # Processa todos os arquivos recursivamente, baixa, extrai e tokeniza.
    print(f"\n=== Iniciando o processamento recursivo de arquivos a partir da Pasta ID: {TARGET_FOLDER_ID} ===")
    tokenized_data = drive_api.drive_service.process_and_embed_all_files_recursively(TARGET_FOLDER_ID)
    print(f"\n=== Processamento de arquivos concluído. ===")

    if tokenized_data:
        print("\nResumo dos arquivos processados e tokenizados:")
        for filename, tokens in tokenized_data.items():
            print(f"- {filename}: {len(tokens)} tokens")
    else:
        print("\nNenhum arquivo foi processado.")

    # Inicia o processamento e geração de embeddings com TensorFlow
    print(f"\n=== Iniciando o processamento recursivo de arquivos e geração de embeddings (TensorFlow) a partir da "
          f"Pasta ID: {TARGET_FOLDER_ID} ===")
    embeddings_data = drive_api.drive_service.process_and_embed_all_files_recursively(TARGET_FOLDER_ID)
    print(f"\n=== Processamento de arquivos e geração de embeddings (TensorFlow) concluídos. ===")

    if embeddings_data:
        print("\nCaminhos para os arquivos de embedding gerados (TensorFlow):")
        for filename, embedding_files in embeddings_data.items():
            print(f"- {filename}:")
            for emb_file in embedding_files:
                print(f"  - {emb_file}")
    else:
        print("\nNenhum embedding foi gerado.")

    # Limpa o diretório temporário
    drive_api.drive_service.cleanup_temp_folder()