import asyncio
import os
import time
import traceback

from langchain_groq import ChatGroq
from pydantic import SecretStr
from transformers import AutoTokenizer

class GroqLLM:
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("A variável de ambiente GROQ_API_KEY não está definida.")
        self.groq_api_key = SecretStr(self.groq_api_key)
        model_tokenizer_id = "meta-llama/Meta-Llama-3-8B"
        try:
            print(f"Carregando tokenizer '{model_tokenizer_id}'")
            self.tokenizer = AutoTokenizer.from_pretrained(model_tokenizer_id)
            print("Tokenizer carregado com sucesso.")
        except Exception as e:
            print(f"Erro ao carregar o tokenizer '{model_tokenizer_id}': {e}")
            print("Certifique-se de que 'transformers' e 'sentencepiece' estão instalados.")
            print("Usando contagem de palavras como fallback (impreciso).")
            # Define como None para indicar que o tokenizer falhou.
            self.tokenizer = None
            raise RuntimeError(f"Não foi possível carregar o tokenizer '{model_tokenizer_id}'.")

        self.llm = ChatGroq(api_key=self.groq_api_key, model="groq/llama3-8b-8192")
        self.tokens_usados_no_minuto = 0
        self.inicio_do_minuto = time.time()
        self.limite_de_tokens_por_minuto = 6000

    def contador_de_tokens(self, texto: str) -> int:
        """Realiza a contagem de tokens apropriada para o modelo Llama 3."""
        if self.tokenizer:
            return len(self.tokenizer.encode(texto))
        else:
            print("Aviso: Tokenizer não carregado. Usando contagem de palavras como fallback (impreciso).")
            return len(texto.split())

    def executar_solicitacao(self, prompt: str) -> str:
        """Executa uma solicitação ao modelo Groq."""
        # Calcula os tokens.
        tokens_na_solicitacao = self.contador_de_tokens(prompt)

        # Se o tokenizer falhou, a contagem de tokens estará incorreta.
        if not self.tokenizer and tokens_na_solicitacao > 0:
            print("Aviso: A contagem de tokens pode estar imprecisa devido à falha no carregamento do tokenizer.")

        # Se necessário esperar, informa o usuário:
        tempo_para_esperar = self.verificar_limite_de_taxa(tokens_na_solicitacao)
        if tempo_para_esperar > 0:
            print(f"Limite de tokens por minuto ({self.limite_de_tokens_por_minuto}) atingido. Esperando"
                  f" {tempo_para_esperar:.1f} segundos.")
            asyncio.sleep(tempo_para_esperar)
            self.inicio_do_minuto = time.time()
            self.tokens_usados_no_minuto = 0

        try:
            # Envia o prompt e recebe a resposta.
            resposta = self.llm.invoke(prompt)
            # Estima os tokens da resposta usando o tokenizer.
            tokens_resposta_estimados = self.contador_de_tokens(resposta.content)
            # Atualiza o contador geral com tokens do prompt + tokens da resposta.
            self.atualizar_contador_de_tokens(tokens_na_solicitacao + tokens_resposta_estimados)  # Atualiza com prompt
            # + resposta
            return resposta.content
        except Exception as e:
            print(f"Erro ao executar solicitação: {e}")
            traceback.print_exc()
            return None

    def verificar_limite_de_taxa(self, tokens_na_solicitacao: int) -> float:):
        """Verifica se o limite de tokens por minuto foi atingido."""
        tempo_atual = time.time()
        tempo_decorrido = tempo_atual - self.inicio_do_minuto

        if tempo_decorrido >= 60:  # Reinicia o contador a cada minuto.
            self.inicio_do_minuto = tempo_atual
            self.tokens_usados_no_minuto = 0
            return 0  # Não há necessidade de esperar.

        # Verifica se a próxima solicitação ultrapassaria o limite
        if (self.tokens_usados_no_minuto + tokens_na_solicitacao) > self.limite_de_tokens_por_minuto:
            # Calcula quanto tempo falta para completar o minuto atual
            tempo_para_esperar = 60.0 - tempo_decorrido
            return tempo_para_esperar
        else:
            # Ainda há espaço no limite para esta solicitação
            return 0

    def atualizar_contador_de_tokens(self, tokens_usados: int):
        """Atualiza o contador de tokens usados no minuto atual."""
        # Garante que o contador não seja atualizado se já passou um minuto.
        tempo_atual = time.time()
        if tempo_atual - self.inicio_do_minuto < 60:
            self.tokens_usados_no_minuto += tokens_usados
        else:
            # Se por acaso passou mais de um minuto entre a verificação e a atualização,
            # reinicia o contador com os tokens atuais.
            self.inicio_do_minuto = tempo_atual
            self.tokens_usados_no_minuto = tokens_usados


def setup_groq_llm():
    """Configura e retorna o modelo de linguagem Groq."""
    try:
        llm_manager = GroqLLM()
        print("LLM Groq configurado com sucesso.")
        return llm_manager
    except ValueError as e:
        print(f"Erro de configuração: {e}")
        return None
    except Exception as e:
        print(f"Ocorreu um erro inesperado ao inicializar o LLM Groq: {e}")
        traceback.print_exc()
        return None