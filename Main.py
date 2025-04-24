import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)

from pathlib import Path

import numpy as np

# Importação dos arquivos existentes
from Authentication import GoogleDriveAPI
from DataBaseManager import DataBaseManager
from EmbeddingGenerator import EmbeddingGenerator
from FaissIndexer import FaissIndexer
from FolderManager import check_directory_existence

# Definição de constantes
TARGET_FOLDER_ID = "1lXQ7R5z8NGV1YGUncVDHntiOFX35r6WO"
TEMP_DOWNLOAD_FOLDER = 'temp_download'
BATCH_SIZE = 4
EMBEDDING_OUTPUT_DIR = 'embeddings_tf'

temp_dir = Path(TEMP_DOWNLOAD_FOLDER)

check_directory_existence(temp_dir)


if __name__ == "__main__":
    drive_api = GoogleDriveAPI()
    drive_service = DataBaseManager(drive_api.service)

    print(f"\n=== Iniciando Listagem Recursiva a partir da Pasta ID: {TARGET_FOLDER_ID} ===")
    all_files_recursive = drive_service.list_files_recursively(folder_id=TARGET_FOLDER_ID)
    print(f"Total de arquivos a processar: {len(all_files_recursive)}")

    # Percorrer all_files_recursive em passos de tamanho BATCH_SIZE. Em cada passo, copia uma fatia em uma nova
    # sublista, sendo salva em file_batches.
    file_batches = [all_files_recursive[i:i + BATCH_SIZE] for i in range(0, len(all_files_recursive), BATCH_SIZE)]

    embedding_generator_instance = EmbeddingGenerator()
    all_embeddings_data = []
    for batch in file_batches:
        batch_result = embedding_generator_instance.process_batch(batch)
        all_embeddings_data.extend(batch_result)

    drive_service.cleanup_temp_folder()
    print("Processamento de todos os arquivos concluído.")
    print(f"Total de embeddings gerados: {len(all_embeddings_data)}")

    # Construção do índice Faiss
    print("\n=== Iniciando a construção do índice ===")

    if all_embeddings_data:
        # Supondo que todos os embeddings tenham a mesma dimensão, pegamos do primeiro.
        # É importante garantir que isso seja verdade no seu fluxo de trabalho.
        try:
            first_embedding_path = all_embeddings_data[0]['embedding_path']
            first_embedding = np.load(first_embedding_path)
            print(f"first_embedding = {first_embedding.shape}")
            embedding_dimension = first_embedding.shape[1]

            faiss_index = FaissIndexer(embedding_dimension)

            if faiss_index.load_and_add_embeddings(all_embeddings_data):
                faiss_index.save_index()
                print("Construção do índice Faiss concluída e salva.")

                # --- Exemplo de como carregar e usar o índice para busca (para teste) ---
                print("\n=== Testando a Busca no Índice Faiss (Exemplo) ===")
                if faiss_index.index is not None:
                    try:
                        if all_embeddings_data:
                            first_embedding_path = all_embeddings_data[0]['embedding_path']
                            first_embedding = np.load(first_embedding_path)
                            query_embedding = first_embedding.reshape(1, -1)
                            k = 3
                            distances, indices = faiss_index.search(query_embedding, top_k=k)
                            print(f"\nResultados da busca para o embedding de exemplo (top {k}):")
                            for i in range(k):
                                if indices[0][i] < len(all_embeddings_data):
                                    result_data = all_embeddings_data[indices[0][i]]
                                    print(f"  - Resultado {i + 1}:")
                                    print(f"    - Distância: {distances[0][i]}")
                                    print(f"    - Nome do Arquivo: {result_data.get('file_name', 'Nome não disponível')}")
                                    print(f"    - ID do Arquivo: {result_data.get('file_id', 'ID não disponível')}")
                                    print(f"    - Caminho do Embedding: {result_data.get('embedding_path','Caminho não disponível')}")
                                    if not all(key in result_data for key in ['file_name', 'file_id', 'embedding_path']):
                                        print(f"    - Dados incompletos: {result_data}")
                                else:
                                    print(f"  - Resultado {i + 1}: Índice fora dos limites dos dados de embedding.")
                        else:
                            print("Aviso: Nenhum embedding disponível para teste de busca.")
                    except Exception as e:
                        print(f"Erro ao realizar busca de exemplo: {e}")
                else:
                    print("O índice Faiss não foi construído corretamente para teste de busca.")
            else:
                print("Falha ao carregar e adicionar os embeddings ao índice.")

        except IndexError:
            print("Nenhum dado de embedding encontrado para determinar a dimensão.")
        except FileNotFoundError:
            print(f"Erro ao carregar o primeiro embedding para determinar a dimensão.")
        except Exception as e:
            print(f"Ocorreu um erro ao inicializar ou carregar os embeddings: {e}")
    else:
        print("Nenhum embedding gerado. Impossível construir o índice Faiss.")
