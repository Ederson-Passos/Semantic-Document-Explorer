import re
import nltk
from nltk.tokenize import word_tokenize

# Verificando se o tokenizador punkt está baixado
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def remove_special_characters(text):
    """Remove caracteres especiais do texto."""
    text = re.sub(r"[^a-zA-Z\s]", "", text) # Mantém apenas letras e espaços
    return text

def convert_to_lowercase(text):
    """Converte o texto para minúsculas."""
    return text.lower()

def tokenize_text(text):
    """Tokeniza o texto em palavras."""
    return word_tokenize(text)

def preprocess_text(text):
    """Realiza o pré-processamento completo do texto."""
    tokens = tokenize_text(convert_to_lowercase(remove_special_characters(text)))
    return tokens
