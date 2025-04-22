import os
import numpy as np
import weaviate
from typing import List, Dict, Any, Optional
from weaviate.exceptions import UnexpectedStatusCodeError
from typing import List, Dict, Any, Optional, Tuple
import datasketch # Testar MinHash, MinHashLSH


class WeaviateIndexer:
    """
    Gerencia a conexão, criação de schema, indexação (incluindo hashes LSH) e busca de
    vetores em uma instância Weaviate.
    """
    def __init__(self,
                 weaviate_url: str,
                 class_name: str,
                 embedding_dimension: int,
                 lsh_num_perm: int = 128,
                 default_properties: Optional[List[Dict[str, Any]]] = None
                 ):
        """
        Inicializa o indexador.
        Args:
            weaviate_url (str): URL da instância Weaviate.
            class_name (str): nome da classe (tabela) a ser usada. Deve começar com letra maiúscula.
            embedding_dimension (int): dimensão dos vetores de embedding que serão indexados.
            lsh_num_perm (int): número de permutações/hiperplanos.
            default_properties (Optional[List[Dict[str, Any]]]): lista de dicionários definindo propriedades adicionais
                                                                 a serem armazenadas com cada vetor.
        """
        if not class_name[0].isupper():
            raise ValueError("Weaviate class name must start with an uppercase letter.")

        self.weaviate_url = weaviate_url
        self.class_name = class_name
        self.embedding_dimension = embedding_dimension
        self.lsh_num_perm = lsh_num_perm
        self._client_config = {'url': self.weaviate_url}

        # Configuração do cliente Weaviate
        try:
            self.client = weaviate.connect_to_local(**self._client_config)
            if not self.client.is_ready():
                raise ConnectionError(f"Não foi possível conectar ou a instância Weaviate em {self.weaviate_url} não "
                                      f"está pronta.")
            print(f"Conectado com sucesso à instância Weaviate em {self.weaviate_url}")
        except Exception as e:
            print(f"Erro ao conectar ao Weaviate em {self.weaviate_url}: {e}")
            raise

        # Definição das propriedades do Schema
        if default_properties is None:
            # Propriedades padrão que queremos armazenar junto com o vetor.
            self.properties = [
                {'name': 'file-name', 'dataType': ['text'], 'description': 'Nome do arquivo original'},
                {'name': 'chunk_id', 'dataType': ['int'], 'description': 'ID do chunk dentro do arquivo'},
                {'name': 'embedding_path', 'dataType': ['text'], 'description': 'Caminho para o arquivo .npy do '
                                                                                'embedding original'},
                # Propriedade para armazenar o hash LSH como string.
                {'name': 'lsh_hash', 'dataType': ['string'], 'description': 'Hash de similaridade do vetor de '
                                                                            'embedding'}
            ]
            print(f"Usando propriedades padrão: {[prop['name'] for prop in self.properties]}")
        else:
            # Validação básica das propriedades fornecidas.
            if not any(prop['name'] == 'lsh_hash' for prop in default_properties):
                raise ValueError("A propriedade 'lsh_hash' (dataType ['string']) é necessária nas propriedades "
                                 "fornecidas.")
            self.properties = default_properties

        # Garante que o schema (classe) exista no Weaviate.
        self._ensure_schema()

    def _ensure_schema(self):
        """
        Verifica se a classe definida existe no Weaviate e a cria se necessário.
        """
        try:
            # Tenta obter o schema da classe
            existing_schema = self.client.collections.get(self.class_name)
            print(f"Schema para a classe '{self.class_name}' já existe.")
            # Validar se o schema existente corresponde ao esperado.
            # Implementar: (ex: verificar propriedades, configuração do vetorizador)
            # Se houver incompatibilidade, pode ser necessário deletar e recriar
            # ou lançar um erro mais específico.
        except weaviate.exceptions.UnexpectedStatusCodeError as e:
            if e.status_code == 404:  # Not Found - Classe não existe
                print(f"Schema para a classe '{self.class_name}' não encontrado. Criando...")
                collection_schema = {
                    'name': self.class_name,
                    'description': f"Armazena chunks de documentos com seus embeddings ({self.embedding_dimension}d) "
                                   f"e hashes LSH.",
                    # Nós forneceremos os embeddings.
                    'vectorizer': "none",
                    'vectorIndexConfig': {
                        'hnsw': {  # Especifica o algoritmo de indexação vetorial.
                            # Parâmetros podem ser ajustados para performance vs custo.
                            'efConstruction': 128,  # Qualidade da construção do índice.
                            'maxConections': 32,  # Número máximo de conexões por nó.
                            'distance': 'cossine'  # Comum para embeddings de texto.
                        }
                    },
                    'properties': self.properties
                }
                try:
                    self.client.collections.create(
                        name=self.class_name,
                        schema=collection_schema
                    )
                    print(f"Schema para a classe '{self.class_name}' criado com sucesso.")
                except Exception as e:
                    print(f"Erro ao criar o schema para a classe {self.class_name}: {e}")
                    raise e
            else:
                # Outro erro inesperado ao tentar obter o schema.
                print(f"Erro inesperado ao verificar/criar o schema '{self.class_name}': {e}")
                raise e
        except Exception as e:
            print(f"Erro geral ao garantir o schema '{self.class_name}': {e}")
            raise e

    # Métodos para geração de Hash LSH
    def _generate_lsh_hash(self, embedding: np.ndarray) -> str:
        pass