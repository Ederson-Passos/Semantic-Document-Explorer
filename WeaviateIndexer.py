import os
import numpy as np
import weaviate
from typing import List, Dict, Any, Optional
from weaviate.exceptions import UnexpectedStatusCodeError
from typing import List, Dict, Any, Optional, Tuple
from datasketch import MinHash, MinHashLSH


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
                 lsh_threshold: int = 0.5,
                 default_properties: Optional[List[Dict[str, Any]]] = None
                 ):
        """
        Inicializa o indexador.
        Args:
            weaviate_url (str): URL da instância Weaviate.
            class_name (str): nome da classe (tabela) a ser usada. Deve começar com letra maiúscula.
            embedding_dimension (int): dimensão dos vetores de embedding que serão indexados.
            lsh_num_perm (int): número de permutações/hiperplanos.
            lsh_threshold (int): define o nível mínimo de similaridade de Jaccard estimado.
            default_properties (Optional[List[Dict[str, Any]]]): lista de dicionários definindo propriedades adicionais
                                                                 a serem armazenadas com cada vetor.
        """
        if not class_name[0].isupper():
            raise ValueError("Weaviate class name must start with an uppercase letter.")

        self.weaviate_url = weaviate_url
        self.class_name = class_name
        self.embedding_dimension = embedding_dimension
        self.lsh_num_perm = lsh_num_perm
        self.lsh_threshold = lsh_threshold
        self.lsh = MinHashLSH(threshold=lsh_threshold, num_perm=lsh_num_perm)
        self._client_config = {'url': self.weaviate_url}

        # Configuração do cliente Weaviate
        try:
            self.client = weaviate.WeaviateClient(**self._client_config)
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

    def _validate_existing_schema(self, existing_schema: Dict[str, Any], expected_properties: Dict[str, Any]):
        """
        Valida o schema existente com o esperado.
        Args:
            existing_schema: O schema existente retornado pelo Weaviate.
            expected_properties: As propriedades esperadas.
        Raises:
            ValueError: Se houver incompatibilidade no schema.
        """
        existing_properties = {prop['name']: prop for prop in existing_schema['properties']}
        if set(expected_properties.keys()) != set(existing_properties.keys()):
            raise ValueError(f"Incompatibilidade no schema da classe '{self.class_name}'. "
                             f"Propriedades esperadas: {set(expected_properties.keys())}, "
                             f"encontradas: {set(existing_properties.keys())}.")

        for prop_name, expected_prop in expected_properties.items():
            if existing_prop := existing_properties.get(prop_name):
                if expected_prop['dataType'] != existing_prop['dataType']:
                    raise ValueError(f"Incompatibilidade no tipo de dados da propriedade '{prop_name}' na "
                                     f"classe '{self.class_name}'.  Esperado: {expected_prop['dataType']}, "
                                     f"encontrado: {existing_prop['dataType']}.")
            else:
                raise ValueError(f"Propriedade '{prop_name}' esperada não encontrada no schema existente da"
                                 f" classe '{self.class_name}'.")

    def _validate_vector_index_config(self, existing_schema: Dict[str, Any], expected_vector_index_config: Dict[str, Any]):
        """
        Valida a configuração do índice vetorial existente com o esperado.
        Args:
            existing_schema: O schema existente retornado pelo Weaviate.
            expected_vector_index_config: A configuração do índice vetorial esperada.
        Raises:
            ValueError: Se houver incompatibilidade na configuração do índice vetorial.
        """
        existing_vector_index_config = existing_schema.get('vectorIndexConfig')

        if not existing_vector_index_config or 'hnsw' not in existing_vector_index_config:
            raise ValueError(f"Incompatibilidade na configuração do índice vetorial da classe "
                             f"'{self.class_name}'. Esperado encontrar a configuração HNSW.")

        for key, value in expected_vector_index_config['hnsw'].items():
            if existing_vector_index_config['hnsw'].get(key) != value:
                raise ValueError(f"Incompatibilidade na configuração HNSW da classe '{self.class_name}'. "
                                 f"Esperado '{key}': {value}, encontrado: "
                                 f"{existing_vector_index_config['hnsw'].get(key)}.")

    def _ensure_schema(self):
        """
        Verifica se a classe definida existe no Weaviate e a cria se necessário.
        Valida o schema existente com o esperado e levanta um erro em caso de incompatibilidade.
        """
        expected_properties = {prop['name']: prop for prop in self.properties}
        expected_vector_index_config = {
            'hnsw': {
                'efConstruction': 128,
                'maxConections': 32,
                'distance': 'cossine'
            }
        }

        try:
            # Tenta obter o schema da classe
            existing_collection = self.client.collections.get(self.class_name)
            if existing_collection is not None:
                print(f"Schema para a classe '{self.class_name}' já existe. Validando...")

                existing_schema = existing_collection.config.get()

                # Validação do schema existente com o esperado
                self._validate_existing_schema(existing_schema, expected_properties)

                # Validação da configuração do índice vetorial
                self._validate_vector_index_config(existing_schema, expected_vector_index_config)

                print(f"Schema para a classe '{self.class_name}' validado com sucesso.")

            else:
                # Classe não existe, então é criada.
                print(f"Schema para a classe '{self.class_name}' não encontrado. Criando...")
                collection_schema = {
                    'name': self.class_name,
                    'description': f"Armazena chunks de documentos com seus embeddings ({self.embedding_dimension}d) ) "
                                   f"e hashes LSH.",
                    'vectorizer': "none",
                    'vectorIndexConfig': {
                        'hnsw': expected_vector_index_config['hnsw']
                    },
                    'properties': self.properties
                }
                try:
                    self.client.collections.create_from_dict(collection_schema)
                    print(f"Schema para a classe '{self.class_name}' criado com sucesso.")
                except Exception as e:
                    print(f"Erro ao criar o schema para a classe {self.class_name}: {e}")
                    raise e

        except weaviate.exceptions.UnexpectedStatusCodeError as e:
            if e.status_code == 404:  # Not Found - Classe não existe
                pass
            else:
                # Outro erro inesperado ao tentar obter o schema.
                print(f"Erro inesperado ao verificar/criar o schema '{self.class_name}': {e}")
                raise e
        except Exception as e:
            print(f"Erro geral ao garantir o schema '{self.class_name}': {e}")
            raise e

    # Métodos para geração de Hash LSH
    def _generate_lsh_hash(self, embedding: np.ndarray) -> str:
        """
        Gera uma chave Minhash LSH a partir de um vetor de embedding.
        Args:
            embedding (np.ndarray): vetor de embedding (1D array).
        Returns:
            str: string representando a chave Minhash LSH.
        """
        if embedding.ndim != 1:
            if embedding.ndim == 2 and embedding.shape[0] ==  1:
                embedding = embedding.flatten()
            else:
                raise ValueError(f"Embedding deve ser um array 1D para MinHash, mas tem shape {embedding.shape}")

        m = MinHash(num_perm=self.lsh_num_perm)
        for i in range(embedding.shape[0]):  # Itera pelo índices do embedding.
            val = embedding[i]
            # Adicionando o valor ao hash.
            for _ in range(int(abs(val) * 100) + 1):  # Multiplica para ter mais granularidade.
                m.update(str(i).encode('utf8'))

        return m.digest().hex()