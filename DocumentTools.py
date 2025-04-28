"""Contém as ferramentas que os agentes usarão para interagir com os arquivos."""
import traceback
import os
import math
from typing import Optional, Any

from crewai.tools import BaseTool
from TextExtractor import extract_text
from transformers import pipeline, AutoTokenizer


class ExtractTextTool(BaseTool):
    name: str = "extract_text_from_file"
    description: str = "Extracts text content from a file. Supports .pdf, .docx, .txt, .xlsx, .pptx, .doc"

    def _run(self, file_path: str) -> str:
        return extract_text(file_path)

class CountWordsTool(BaseTool):
    name: str = "count_words"
    description: str = "Counts the number of words in a given text."

    def _run(self, text: str) -> int:
        return len(text.split())

class HuggingFaceSummarizationTool(BaseTool):
    name: str = "generate_concise_summary"
    description: str = (
        "Generates a concise summary of a document's content using a Hugging Face model. "
        "Handles potentially large documents by processing them in chunks."
    )
    model_name: str = "facebook/bart-large-cnn"
    tokenizer: Optional[Any] = None
    summarizer: Optional[Any] = None

    def __init__(self, model_name: str = "facebook/bart-large-cnn", **kwargs):
        super().__init__(**kwargs)
        try:
            print(f"Initializing HuggingFaceSummarizationTool with model: {model_name}")
            # Carrega o tokenizer e o pipeline de sumarização uma vez na inicialização.
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.summarizer = pipeline("summarization", model=model_name, tokenizer=self.tokenizer)
            print("Summarization pipeline loaded successfully.")
        except Exception as e:
            print(f"Error initializing summarization pipeline for model {model_name}: {e}")
            print("Summarization tool may not function correctly.")
            # Considerar levantar um erro ou ter um fallback mais robusto
            self.summarizer = None
            self.tokenizer = None

    def _run(self, file_path: str, max_summary_length: int = 150, min_summary_length: int = 30) -> str:
        """
        Summarizes the text extracted from the file_path. Handles large documents via chunking.
        Args:
            file_path (str): The path to the file to summarize.
            max_summary_length (int): The maximum number of tokens for the summary.
            min_summary_length (int): The minimum number of tokens for the summary.
        Returns:
            str: The generated summary or an error message.
        """
        if not self.summarizer or not self.tokenizer:
            return "Error: Summarization model not initialized."

        print(f"Attempting to summarize file: {file_path}")
        try:
            # 1. Extrair texto do documento
            text = extract_text(file_path)
            if not text or not text.strip():
                return f"No text could be extracted from '{os.path.basename(file_path)}'."

            # 2. Lidar com Documentos Grandes (Chunking)
            # Obter o limite máximo de tokens do modelo (menos espaço para tokens especiais)
            model_max_length = self.tokenizer.model_max_length
            # Definir um tamanho de chunk um pouco menor para segurança
            chunk_size = model_max_length - 50
            # Definir uma sobreposição para manter o contexto entre chunks
            overlap = 50

            # Tokenizar o texto completo para saber o número total de tokens
            tokens = self.tokenizer.encode(text)
            num_tokens = len(tokens)
            print(f"  - Total tokens in document: {num_tokens}")

            if num_tokens <= chunk_size:
                # Se o texto couber em um único chunk, sumariza diretamente
                print(f"  - Document fits in one chunk. Summarizing directly...")
                summary_list = self.summarizer(
                    text,
                    max_length=max_summary_length,
                    min_length=min_summary_length,
                    do_sample=False
                )
                summary = summary_list[0]['summary_text']

            else:
                # Se for maior, processa em chunks
                print(f"  - Document exceeds model limit ({model_max_length} tokens). Processing in chunks...")
                all_summaries = []
                start = 0
                while start < num_tokens:
                    end = min(start + chunk_size, num_tokens)
                    # Decodifica o chunk de volta para texto
                    chunk_text = self.tokenizer.decode(tokens[start:end], skip_special_tokens=True)

                    if not chunk_text.strip():  # Pula chunks vazios
                        start += chunk_size - overlap
                        continue

                    print(f"    - Summarizing chunk: tokens {start} to {end}")
                    try:
                        # Ajusta o tamanho do resumo por chunk (proporcional ao número de chunks)
                        num_chunks = math.ceil(num_tokens / (chunk_size - overlap))
                        chunk_max_len = max(min_summary_length //
                                            num_chunks + 10, max_summary_length //
                                            num_chunks + 10)  # Ajuste heurístico
                        chunk_min_len = max(10, min_summary_length // num_chunks)

                        chunk_summary_list = self.summarizer(
                            chunk_text,
                            max_length=chunk_max_len,
                            min_length=chunk_min_len,
                            do_sample=False
                        )
                        all_summaries.append(chunk_summary_list[0]['summary_text'])
                    except Exception as chunk_e:
                        print(f"    - Error summarizing chunk {start}-{end}: {chunk_e}")
                        # Pode adicionar um placeholder ou pular o chunk
                        all_summaries.append(f"[Error summarizing chunk {start}-{end}]")

                    # Avança para o próximo chunk com sobreposição
                    start += chunk_size - overlap

                # Combina os resumos dos chunks (pode precisar de sumarização adicional se for muito longo)
                combined_summary = "\n".join(all_summaries)
                print(f"  - Combined summaries from {len(all_summaries)} chunks.")

                # Opcional: Sumarizar o resumo combinado se ele for muito longo
                combined_tokens = self.tokenizer.encode(combined_summary)
                if len(combined_tokens) > model_max_length:
                    print("  - Combined summary is too long. Summarizing the combined summary...")
                    final_summary_list = self.summarizer(
                        combined_summary,
                        max_length=max_summary_length,
                        min_length=min_summary_length,
                        do_sample=False
                    )
                    summary = final_summary_list[0]['summary_text']
                else:
                    summary = combined_summary


            print(f"  - Summary generated for: {os.path.basename(file_path)}")
            return summary.strip()

        except FileNotFoundError:
            return f"Error: File not found at '{file_path}'"
        except Exception as e:
            print(f"An unexpected error occurred during summarization for {file_path}: {e}")
            traceback.print_exc()
            return f"Error summarizing file '{os.path.basename(file_path)}': {e}"