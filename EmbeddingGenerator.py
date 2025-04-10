import os
import numpy as np
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf

class EmbeddingGenerator:
    def __init__(self, model_name='bert-base-uncased', batch_size=32, output_dir='embeddings_tf'):
        """
        Inicializa o gerador de embeddings com o modelo TensorFlow, gerando embeddings a partir de tokens.
        Args:
            model_name (str): O nome do modelo Transformer pré-treinado a ser usado.
            batch_size (int): O número de sequências a serem processadas por lote.
            output_dir (str): O diretório onde os vetores de embedding serão salvos.
        """
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = TFBertModel.from_pretrained(model_name)
        self.batch_size = batch_size
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Verifica se a GPU está disponível e a usa, caso contrário usa a CPU
        if tf.config.list_physical_devices('GPU'):
            print("GPU encontrada, usando para geração de embeddings.")
            self.device = '/GPU:0'
        else:
            print("Nenhuma GPU encontrada, usando CPU para geração de embeddings.")
            self.device = '/CPU:0'

    def generate_embeddings(self, tokens_list, filename_prefix="document"):
        """
        Gera embeddings para uma lista de tokens, processando-os em lotes usando TensorFlow e a device configurada.
        Args:
            tokens_list (list): Uma lista de tokens.
            filename_prefix (str): Prefixo para o nome do arquivo onde os embeddings serão salvos.
        Returns:
            str: O caminho para o arquivo onde os embeddings foram salvos.
        """
        all_embeddings = []
        for i in range(0, len(tokens_list), self.batch_size):
            batch_tokens = tokens_list[i:i + self.batch_size]

            # Adiciona os tokens especiais [CLS] no início e [SEP] no final de cada sequência
            batch_tokens_with_special_tokens = ['[CLS]'] + batch_tokens + ['[SEP]']

            # Tokeniza e converte os tokens para IDs
            inputs = self.tokenizer(batch_tokens_with_special_tokens,
                                    return_tensors="tf",
                                    padding=True,
                                    truncation=True,
                                    max_length=512)

            # Envolve a inferência do modelo no contexto do dispositivo.
            with tf.device(self.device):
                outputs = self.model(**inputs, output_hidden_states=True)
                # O último hidden state é uma sequência de vetores para cada token.
                # Pega a representação do token [CLS] como a representação da sequência.
                embeddings = outputs.hidden_states[-1][:, 0, :].numpy() # (batch_size, embedding_dimension)
            all_embeddings.extend(embeddings)

        output_filename = os.path.join(self.output_dir, f"{filename_prefix}_embeddings.npy")
        np.save(output_filename, np.array(all_embeddings))
        print(f"Embeddings para '{filename_prefix}' salvos em: {output_filename}")
        return output_filename

    def process_batch(self, batch_files):
        """
        Processa um lote de arquivos, extrai texto, tokeniza e gera embeddings.
        É executada para cada processo no pool de multiprocessamento.
        """
        from Authentication import GoogleDriveAPI
        from TextExtractor import process_and_tokenize_file, TEMP_DOWNLOAD_FOLDER
        import os
        import io
        from googleapiclient.http import MediaIoBaseDownload
        from googleapiclient.errors import HttpError
        import multiprocessing as mp

        CHUNK_SIZE = 310
        drive_api = GoogleDriveAPI()
        embeddings_data = []

        # A instância de EmbeddingGenerator é criada dentro desta função para cada processo filho.
        embedding_generator = EmbeddingGenerator()

        for file_info in batch_files:
            file_id = file_info.get('id')
            file_name = file_info.get('name')

            if not file_id or not file_name:
                print(f"[Processo {mp.current_process().pid}] Informação de arquivo inválida: {file_info}")
                continue

            download_path = os.path.join(TEMP_DOWNLOAD_FOLDER, file_name)

            print(f"[Processo {mp.current_process().pid}] Iniciando tentativa de download do arquivo '{file_name}' (ID: {file_id}).")
            try:
                request = drive_api.service.files().get_media(fileId=file_id)
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                    if status:
                        print(f"[Processo {mp.current_process().pid}] Download '{file_name}': {int(status.progress() * 100)}%...", end='')
                print(f"[Processo {mp.current_process().pid}] Download de '{file_name}' concluído.")
                with open(download_path, "wb") as f:
                    f.write(fh.getvalue())

                print(f"[Processo {mp.current_process().pid}] Arquivo '{file_name}' baixado para '{download_path}'.")
                _, tokens = process_and_tokenize_file(download_path)

                if tokens:
                    num_chunks = (len(tokens) + CHUNK_SIZE - 1) // CHUNK_SIZE
                    for i in range(num_chunks):
                        start_index = i * CHUNK_SIZE
                        end_index = min((i + 1) * CHUNK_SIZE, len(tokens))
                        chunk_tokens = tokens[start_index:end_index]
                        if chunk_tokens:
                            embedding_filename = f"{os.path.splitext(file_name)[0]}_part_{i}"
                            # Chama o método generate_embeddings da instância LOCAL.
                            embedding_path = embedding_generator.generate_embeddings(chunk_tokens, embedding_filename)
                            embeddings_data.append({
                                "filename": file_name,
                                "chunk_id": i,
                                "embedding_path": embedding_path
                            })
                if os.path.exists(download_path):
                    os.remove(download_path)
                    print(f"[Processo {mp.current_process().pid}] Arquivo temporário '{download_path}' removido.")
                else:
                    print(f"[Processo {mp.current_process().pid}] Arquivo temporário '{download_path}' não encontrado e não pôde ser removido.")

            except HttpError as error:
                print(f"[Processo {mp.current_process().pid}] Erro ao baixar o arquivo '{file_name}' (ID: {file_id}): {error}")
            except Exception as e:
                print(f"[Processo {mp.current_process().pid}] Erro inesperado ao processar o arquivo '{file_name}' (ID: {file_id}): {e}")

        return embeddings_data