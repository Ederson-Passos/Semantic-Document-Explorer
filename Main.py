import multiprocessing as mp
from Authentication import GoogleDriveAPI
from DataBaseManager import DataBaseManager
from EmbeddingGenerator import EmbeddingGenerator

TARGET_FOLDER_ID = "1yAEcnT_ecmPpjgkBSUeocA86tBP5D4rb"
BATCH_SIZE = 4
NUM_PROCESSES = mp.cpu_count()

def process_batch_wrapper(batch_files):
    # Instancia o EmbeddingGenerator dentro do processo filho
    embedding_generator_instance = EmbeddingGenerator()
    return embedding_generator_instance.process_batch(batch_files)

if __name__ == "__main__":
    drive_api = GoogleDriveAPI()
    drive_service = DataBaseManager(drive_api.service)

    print(f"\n=== Iniciando Listagem Recursiva a partir da Pasta ID: {TARGET_FOLDER_ID} ===")
    all_files_recursive = drive_service.list_files_recursively(folder_id=TARGET_FOLDER_ID)
    print(f"Total de arquivos a processar: {len(all_files_recursive)}")

    file_batches = [all_files_recursive[i:i + BATCH_SIZE] for i in range(0, len(all_files_recursive), BATCH_SIZE)]

    all_embeddings_data = []
    with mp.Pool(processes=NUM_PROCESSES) as pool:
        results = pool.map(process_batch_wrapper, file_batches) # Passa apenas file_batches
        for batch_result in results:
            all_embeddings_data.extend(batch_result)

    drive_service.cleanup_temp_folder()
    print("Processamento de todos os arquivos concluído.")
    print(f"Total de embeddings gerados: {len(all_embeddings_data)}")
