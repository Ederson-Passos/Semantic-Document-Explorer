import os
import time
import traceback
from typing import Optional

from langchain_groq import ChatGroq
from pydantic import SecretStr
from transformers import AutoTokenizer

class GroqLLM:
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("A variável de ambiente GROQ_API_KEY não está definida.")
        self.groq_api_key = SecretStr(self.groq_api_key)

        # Tenta carregar o token do Hugging Face
        hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
        if not hf_token:
            print("Aviso: Variável de ambiente HUGGING_FACE_HUB_TOKEN não definida.")
            print("O carregamento do tokenizer pode falhar se o modelo for restrito (gated).")
            raise ValueError("HUGGING_FACE_HUB_TOKEN é necessário para modelos restritos.")


        model_tokenizer_id = "meta-llama/Meta-Llama-3-8B"
        try:
            print(f"Carregando tokenizer '{model_tokenizer_id}'...")
            # Passa o token para from_pretrained
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_tokenizer_id,
                token=hf_token
            )
            print("Tokenizer carregado com sucesso.")
        except Exception as e:
            print(f"Erro ao carregar o tokenizer '{model_tokenizer_id}': {e}")
            print("Verifique se:")
            print("  1. Você aceitou os termos de uso em https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct")
            print("  2. A variável de ambiente HUGGING_FACE_HUB_TOKEN está definida corretamente no seu .env "
                  "ou sistema.")
            print("  3. As bibliotecas 'transformers' e 'sentencepiece' estão instaladas.")
            print("Usando contagem de palavras como fallback (impreciso).")
            self.tokenizer = None
            # Mantém o erro fatal, pois a contagem de tokens é importante
            raise RuntimeError(f"Não foi possível carregar o tokenizer '{model_tokenizer_id}'. "
                               f"Verifique o acesso e autenticação.")

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

    def truncate_content(self, content: str, current_tokens: int, max_tokens: int, batch_number: int) -> str:
        """
        Trunca o conteúdo se a contagem de tokens exceder o máximo permitido.
        Args:
            content: O texto completo do arquivo.
            current_tokens: A contagem de tokens já calculada para o conteúdo.
            max_tokens: O número máximo de tokens que a LLM pode processar.
            batch_number: O número do lote atual (para fins de log).
        Returns:
            O conteúdo original ou uma versão truncada com um aviso.
        """
        if current_tokens > max_tokens:
            print(f"      [Lote {batch_number}] ALERTA: Conteúdo excedeu o limite de tokens "
                  f"({current_tokens} > {max_tokens}). Truncando...")

            # --- Estratégia de Truncamento ---
            if self.tokenizer:
                # Estratégia preferencial: usar o tokenizer para truncar
                try:
                    # Codifica, pega os primeiros 'max_tokens' e decodifica de volta
                    # Adiciona uma pequena margem de segurança (ex: max_tokens - 10)
                    safe_max_tokens = max(10, max_tokens - 10) # Garante que não seja negativo
                    token_ids = self.tokenizer.encode(content, add_special_tokens=False) # Não adiciona BOS/EOS aqui
                    truncated_ids = token_ids[:safe_max_tokens]
                    # Decodifica, skip_special_tokens=True para evitar adicionar tokens especiais automaticamente
                    truncated_content = self.tokenizer.decode(truncated_ids, skip_special_tokens=True)
                    print(f"      [Lote {batch_number}] Conteúdo truncado usando tokenizer para ~{len(truncated_ids)} tokens.")
                except Exception as e:
                    print(f"      [Lote {batch_number}] Erro ao truncar com tokenizer ({e}), usando fallback de caracteres.")
                    # Fallback para caracteres se o tokenizer falhar
                    estimated_chars_per_token = len(content) / current_tokens if current_tokens > 0 else 4
                    max_chars = int(max_tokens * estimated_chars_per_token * 0.95) # 95% para segurança
                    truncated_content = content[:max_chars]
            else:
                # Fallback se o tokenizer não carregou: truncar por caracteres
                print(f"      [Lote {batch_number}] Tokenizer não disponível, truncando por caracteres.")
                # Estima quantos caracteres por token, em média (ajuste se necessário)
                # Usar a estimativa de current_tokens se disponível, senão um valor padrão
                estimated_chars_per_token = len(content) / current_tokens if current_tokens > 0 else 4
                # Calcula o número máximo de caracteres, com uma pequena margem de segurança
                max_chars = int(max_tokens * estimated_chars_per_token * 0.95) # 95% para segurança
                truncated_content = content[:max_chars]

            # Adiciona um aviso claro no final do conteúdo truncado
            truncation_warning = "\n\n[... CONTEÚDO TRUNCADO DEVIDO AO LIMITE DE TOKENS ...]"
            # Garante que o aviso caiba, removendo parte do final se necessário
            if len(truncated_content) + len(truncation_warning) > len(content): # Evita crescer o conteúdo
                truncated_content = truncated_content[:len(content)-len(truncation_warning)-1]

            truncated_content += truncation_warning

            # Opcional: Recalcular tokens do conteúdo truncado para log
            # new_token_count = self.contador_de_tokens(truncated_content)
            # print(f"      [Lote {batch_number}] Conteúdo truncado para aproximadamente {new_token_count} tokens.")

            return truncated_content
        else:
            # Se não excedeu o limite, retorna o conteúdo original
            return content

    def executar_solicitacao(self, prompt: str) -> Optional[str]:
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
            time.sleep(tempo_para_esperar)
            self.inicio_do_minuto = time.time()
            self.tokens_usados_no_minuto = 0

        try:
            print(f"Enviando prompt para Groq ({tokens_na_solicitacao} tokens estimados)...")
            resposta = self.llm.invoke(prompt)
            print("Resposta recebida do Groq.")

            # Verifica se a resposta e o conteúdo são válidos
            if resposta and isinstance(resposta.content, str):
                response_content = resposta.content
                # Estima os tokens da resposta
                tokens_resposta_estimados = self.contador_de_tokens(response_content)
                print(f"Resposta contém {tokens_resposta_estimados} tokens estimados.")
                # Atualiza o contador geral com tokens do prompt + tokens da resposta
                self.atualizar_contador_de_tokens(tokens_na_solicitacao + tokens_resposta_estimados)
                return response_content # Retorna o conteúdo da string
            else:
                # Caso onde resposta.content não é uma string (pode ser None, etc.)
                print(f"Aviso: A resposta do LLM não continha conteúdo de texto válido ou foi None. Resposta: {resposta}")
                # Decide o que fazer - retornar None é consistente com o bloco except
                return None

        except Exception as e:
            print(f"Erro ao executar solicitação: {e}")
            traceback.print_exc()
            return None

    def verificar_limite_de_taxa(self, tokens_na_solicitacao: int) -> float:
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
            return 0.0

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

def initialize_llm():
    """Inicializa o LLM Manager e retorna a instância do ChatGroq LLM."""
    print("Inicializando o LLM Manager...")
    llm_manager = setup_groq_llm()
    if llm_manager is None:
        print("Falha ao inicializar o LLM Manager. Encerrando.")
        return None, None

    try:
        if hasattr(llm_manager, 'llm') and llm_manager.llm:
            groq_chat_llm = llm_manager.llm
            print("Instância do ChatGroq LLM obtida com sucesso.")
            return llm_manager, groq_chat_llm
        else:
            print("Erro: O LLM Manager foi inicializado, mas a instância do LLM (ChatGroq) não está disponível.")
            return None, None
    except AttributeError:
        print("Erro: O objeto retornado por setup_groq_llm() não possui o atributo 'llm'.")
        return None, None