"""
Contém a lógica de indexação com Faiss (Facebook AI Similarity Search), biblioteca de código aberto
projetada para fornecer um mecanismo de busca de similaridade e agrupamento de vetores densos de alta dimensão.
"""
import os
import numpy as np
import faiss
from typing import List


class FaissIndexer:
    def __init__(self,
                 emdedding_dimension: int,
                 index_type: str = 'IndexFlatL2',  # Verificar outras opções
                 index_path: str = 'document_embeddings.faiss'):
        """
        Inicializa o indexador Faiss.
        Args:
            emdedding_dimension (int): Dimensão dos vetores que serão indexados. Deve corresponder
                                       à dimensão da saída do modelo Transformer.
            index_type (str): Tipo de índice a ser criado (array). 'IndexFlatL2' é uma busca mais exata, mas lenta
                              para grandes datasets, onde 'L2' representa a distância euclidiana.
            index_path (str): Caminho onde o índice Faiss será salvo e de onde poderá ser carregado.
        """
        self.emdedding_dimension = emdedding_dimension
        self.index_type = index_type
        self.index_path = index_path
        self.index = self._create_index()
        self.is_trained = False  # Flag para sinalizar se o índice precisa de treinamento.

        print(f"Índice Faiss do tipo '{self.index_type}' inicializado para"
              f"embeddings de dimensão {self.emdedding_dimension}.")

    def _check_index_initialized(self):
        """
        Verifica se o índice Faiss foi inicializado.
        Raises:
            RuntimeError: Se o índice não foi inicializado.
        """
        if self.index is None:
            raise RuntimeError("O índice Faiss não foi inicializado.")

    def _create_index(self) -> faiss.Index:
        """
        Cria o índice Faiss com base no tipo especificado.
        Returns:
            faiss.Index: objeto de índice Faiss criado.
        """
        if self.index_type == 'IndexFlatL2':
            # Busca Nearest Neighbors por força bruta usando distância euclidiana (L2).
            # Não requer treinamento porque compara todos os vetores diretamente.
            index = faiss.IndexFlatL2(self.emdedding_dimension)
        else:
            # Possível adicionar outros tipos de índices neste trecho, dependendo do
            # dataset e dos requisitos de performance.
            raise ValueError(f"Tipo de índice '{self.index_type}' não suportado atualmente.")
        return index

    def add_embeddings(self, embeddings: np.ndarray):
        """
        Adiciona um lote de vetores de embedding ao índice Faiss.
        Args:
             embeddings (np.ndarray): matriz numpy onde cada linha é um vetor de embedding e cada coluna é o
                                      tamanho do vetor. A forma da matriz é (num_embeddings, embedding_dimension).
        """
        self._check_index_initialized()

        # Verifica se a dimensão dos embeddings corresponde à dimensão esperada do índice.
        if embeddings.shape[1] != self.emdedding_dimension:
            raise ValueError(f"Dimensão do embedding ({embeddings.shape[1]}) não corresponde"
                             f"à dimensão do índice ({self.emdedding_dimension}).")

        # Adiciona os embeddings ao índice.
        self.index.add(embeddings)
        print(f"{embeddings.shape[0]} embeddings adicionados ao índice."
              f"Tamanho atual do índice: {self.index.ntotal} vetores.")

    def train_index(self, embeddings: np.ndarray):
        """
        Treina o índice Faiss com os vetores de embedding fornecidos, se o índice requerer treinamento.
        Args:
            embeddings (np.ndarray): matriz numpy com os vetores de embedding para treinamento.
        """
        self._check_index_initialized()

        if self.index.is_trained:
            print("O índice Faiss já foi treinado.")
            return

        # Verifica se o  índice suporta treinamento.
        if hasattr(self.index, 'train'):
            print("Iniciando o treinamento do índice Faiss...")
            self.index.train(embeddings)
            self.is_trained = True
            print("Treinamento do índice Faiss concluído.")
        else:
            print(f"O tipo de índice '{self.index_type}' não requer treinamento.")

    def save_index(self):
        """
        Salva o índice Faiss para um arquivo.
        """
        self._check_index_initialized()

        faiss.write_index(self.index, self.index_path)
        print(f"Índice Faiss salvo em: {self.index_path}")

    def load_index(self):
        """
        Carrega um índice Faiss de um arquivo.
        """
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            self.emdedding_dimension = self.index.d  # Atualiza a dimensão do embedding ao carregar.
            print(f"Índice Faiss carregado de: {self.index_path} (dimensão: {self.emdedding_dimension}, total de"
                  f"vetores: {self.index.ntotal}).")
        else:
            print(f"Arquivo de índice Faiss não encontrado em: {self.index_path}. Um novo índice precisará ser"
                  f"criado e pulado.")
            self.index = self._create_index()  # Garante que um índice exista mesmo se o arquivo não for encontrado.
            self.is_trained = False

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        """
        Realiza uma busca no índice Faiss para encontrar os k vetores mais similares ao vetor de consulta.
        Args:
            query_embedding (np.ndarray): vetor de embedding da consulta (deve ter a mesma dimensão dos
                                          embeddings no índice.
            top_k (int): número de vizinhos mais próximos a serem retornados.
        Returns:
            tuple contendo:
                - distances (np.ndarray): array de distâncias entre o vetor de consulta e os top_k vizinhos.
                - indices (np.ndarray): array de índices dos top_k vizinhos no índice Faiss.
        """
        self._check_index_initialized()

        # Verificando se está sendo passada uma query com mais de um vetor ou
        # Uma query com vetor de tamanho diferente dos vetores do index.
        if query_embedding.shape[0] != 1 or query_embedding.shape[1] != self.emdedding_dimension:
            raise ValueError(f"Formato do vetor de consulta inválido. Esperado (1, {self.emdedding_dimension}),"
                             f"recebido {query_embedding.shape}.")

        distances, indices = self.index.search(query_embedding, top_k)
        return distances, indices
