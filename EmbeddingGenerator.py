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