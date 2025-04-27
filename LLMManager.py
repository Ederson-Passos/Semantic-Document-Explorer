import os
import traceback

from langchain_groq import ChatGroq
from pydantic import SecretStr


def setup_groq_llm():
    """Configura e retorna o modelo de linguagem Groq."""
    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("A variável de ambiente GROQ_API_KEY não está definida.")

        # Envolver a chave com SecretStr.
        groq_api_key = SecretStr(groq_api_key)

        # Criando uma instância do Groq.
        llm = ChatGroq(
            api_key=groq_api_key,
            model="groq/llama3-8b-8192"
        )
        print("LLM Groq configurado com sucesso.")
        return llm
    except ImportError:
        print("Erro: A biblioteca groq não está instalada.")
        print("Instale usando: pip install groq")
        return None
    except ValueError as e:
        print(f"Erro de configuração: {e}")
        return None
    except Exception as e:
        print(f"Ocorreu um erro inesperado ao inicializar o LLM Groq: {e}")
        traceback.print_exc()
        return None