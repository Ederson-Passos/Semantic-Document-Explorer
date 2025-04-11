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
                 index_type: str = 'IndexFlatL2', # Verificar outras opções
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
        pass