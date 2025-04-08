from Authentication import authenticate, list_files, download_multiple_files
from TextExtractor import extractor
from Tokenization import preprocess_text

# Exemplo de uso
creds = authenticate()
files = list_files(creds, page_size=20)     # Lista os primeiros 20 arquivos
download_multiple_files(creds, files)   # Baixa todos os arquivos listados


extracted_text = extractor(files)
print(extracted_text)

text = "Este é um exemplo de texto, com pontuação! 123 e caracteres especiais @#$."
preprocessed_tokens = preprocess_text(text)
print(f"Texto original: {text}")
print(f"Tokens pré-processados: {preprocessed_tokens}")