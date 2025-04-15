"""
Contém a lógica de indexação com Faiss (Facebook AI Similarity Search), biblioteca de código aberto
projetada para fornecer um mecanismo de busca de similaridade e agrupamento de vetores densos de alta dimensão.
"""
import os
import numpy as np
import faiss
from typing import List, Dict, Any, Optional


class FaissIndexer:
    def __init__(self,
                 embedding_dimension: int,
                 index_type: str = 'IndexFlatL2',  # Verificar outras opções
                 index_path: str = 'document_embeddings.faiss'):
        """
        Inicializa o indexador Faiss.
        Args:
            emdedding_dimension (Optional[int]): Dimensão dos vetores que serão indexados. Deve corresponder
                                                 à dimensão da saída do modelo Transformer.
                                                 Pode ser definida durante o carregamento.
            index_type (str): Tipo de índice a ser criado (array). 'IndexFlatL2' é uma busca mais exata, mas lenta
                              para grandes datasets, onde 'L2' representa a distância euclidiana.
            index_path (str): Caminho onde o índice Faiss será salvo e de onde poderá ser carregado.
        """
        self.embedding_dimension = embedding_dimension
        self.index_type = index_type
        self.index_path = index_path
        self.index = None
        self.is_trained = False  # Flag para sinalizar se o índice precisa de treinamento.

        print(f"Índice Faiss do tipo '{self.index_type}' inicializado.")
        if self.embedding_dimension is not None:
            print(f"Dimensão esperada do embedding: {self.embedding_dimension}")

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
        if self.embedding_dimension is None:
            raise ValueError("A dimensão do embedding deve ser definida antes de criar o índice.")
        if self.index_type == 'IndexFlatL2':
            # Busca Nearest Neighbors por força bruta usando distância euclidiana (L2).
            # Não requer treinamento porque compara todos os vetores diretamente.
            index = faiss.IndexFlatL2(self.embedding_dimension)
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
        if embeddings.shape[1] != self.embedding_dimension:
            raise ValueError(f"Dimensão do embedding ({embeddings.shape[1]}) não corresponde"
                             f"à dimensão do índice ({self.embedding_dimension}).")

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
        Carrega um índice Faiss de um arquivo já existente para a memória.
        """
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            self.embedding_dimension = self.index.d  # Atualiza a dimensão do embedding ao carregar.
            print(f"Índice Faiss carregado de: {self.index_path} (dimensão: {self.embedding_dimension}, total de"
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
        if query_embedding.shape[0] != 1 or query_embedding.shape[1] != self.embedding_dimension:
            raise ValueError(f"Formato do vetor de consulta inválido. Esperado (1, {self.embedding_dimension}),"
                             f"recebido {query_embedding.shape}.")

        distances, indices = self.index.search(query_embedding, top_k)
        return distances, indices

    def load_and_add_embeddings(self, all_embeddings_data: List[Dict[str, Any]]) -> bool:
        """
        Carrega os embeddings dos arquivos especificados e os adiciona ao índice.
        Args:
            all_embeddings_data (List[Dict[str, Any]]): lista de dicionários, onde cada dicionário contém informações
                                                        sobre o embedding.
        Returns:
            bool: True se os embeddings foram carregados e adicionados com sucesso.
        """
        all_embeddings = []
        index_created = False  # Flag para verificar se o índice foi criado

        for embedding_info in all_embeddings_data:
            embedding_path = embedding_info['embedding_path']
            try:
                embedding = np.load(embedding_path)
                current_dimension = embedding.shape[1] if embedding.ndim > 1 else embedding.shape[0]

                if self.embedding_dimension is None:
                    self.embedding_dimension = current_dimension
                    self.index = self._create_index()
                    print(f"Índice Faiss criado com dimensão: {self.embedding_dimension}")
                    index_created = True
                elif current_dimension != self.embedding_dimension:
                    print(f"Erro: embedding carregado de '{embedding_path}' tem dimensão {current_dimension}, que não"
                          f" corresponde à dimensão esperada ({self.embedding_dimension}).")
                    return False

                all_embeddings.append(embedding)

            except FileNotFoundError:
                print(f"Erro: Arquivo de embedding não encontrado em: {embedding_path}")
            except Exception as e:
                print(f"Erro ao carregar embedding de {embedding_path}: {e}")

        if not all_embeddings:
            print("Nenhum embedding carregado. Impossível construir o índice.")
            return False

        if not index_created and self.embedding_dimension is not None:
            self.index = self._create_index()

        # Concatena os arrays ao longo do eixo vertical, resultando na forma (num_arrays, dim_arrays).
        embeddings_array = np.concatenate(all_embeddings, axis=0)
        num_embeddings = embeddings_array.shape[0]
        print(f"Total de embeddings carregados para o índice: "
              f"{num_embeddings} com dimensão: {self.embedding_dimension}.")

        self.add_embeddings(embeddings_array)
        return True
