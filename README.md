# Requisitos

Certifique-se de ter o Python instalado na versão **3.10.x**. Outras versões podem apresentar problemas de compatibilidade com as dependências do projeto. Você pode baixar o Python 3.10 em [https://www.python.org/downloads/release/python-3100/](https://www.python.org/downloads/release/python-3100/).

**Observação:** O projeto foi atualizado para utilizar a API do Gemini para embeddings. As instruções abaixo foram atualizadas para refletir essas mudanças.



## Configuração das Credenciais do Google Drive para Testes

Este aplicativo utiliza a API do Google Drive para acessar arquivos para análise. Para testá-lo com sua própria conta do Google Drive, você precisará realizar os seguintes passos para obter suas próprias credenciais e configurar os arquivos necessários:

1.  **Ativar a API do Google Drive no Google Cloud Platform:**
    * Acesse o [Google Cloud Console](https://console.cloud.google.com/).
    * Selecione ou crie um novo projeto. Dê um nome ao projeto, como "SemanticDocumentExplorer".
    * No menu de navegação, vá em "APIs e serviços" > "Biblioteca".
    * Procure por "Google Drive API" e clique em "Ativar".

2.  **Criar Credenciais OAuth 2.0:**
    * No menu de navegação, vá em "APIs e serviços" > "Credenciais".
    * Clique em "Criar credenciais" e selecione "ID do cliente OAuth".
    * Se você ainda não configurou a tela de consentimento OAuth, será solicitado a fazê-lo:
        * Clique em "Configurar tela de consentimento".
        * Selecione o tipo de usuário ("Externo" para testes com contas fora da sua organização, a menos que seja um aplicativo interno). **Importante:** Para o tipo de usuário "Externo", você precisará adicionar usuários de teste na tela de consentimento para que eles possam usar o aplicativo.
        * Preencha o nome do aplicativo, e-mail de suporte do usuário e outras informações necessárias.
        * Em "Domínios autorizados", você pode deixar em branco para testes locais.
        * Clique em "Salvar".
    * De volta à tela de "Criar credenciais", selecione novamente "ID do cliente OAuth".
    * Selecione o tipo de aplicativo como "Aplicativo para computador".
    * Dê um nome ao seu cliente OAuth (por exemplo, "SemanticDocumentExplorer Test").
    * Clique em "Criar".
    * Uma janela modal aparecerá com seu "ID do cliente" e "Secreto do cliente". **Clique em "Fazer o download do JSON" (ou um botão similar).** Este arquivo JSON contém suas credenciais.

3.  **Salvar o arquivo de credenciais:**
    * Salve o arquivo JSON que você baixou (geralmente com um nome como `credentials.json`) **na raiz do seu projeto**, exatamente como você tem o seu arquivo `credentials.json` localmente. **É crucial que este arquivo não seja incluído no repositório Git devido às informações sensíveis.**

4.  **Arquivo de token (gerado automaticamente):**
    * A primeira vez que o aplicativo for executado com as credenciais corretas, ele tentará se autenticar com o Google Drive. Se a autenticação for bem-sucedida, um arquivo de token (geralmente `token.json`) será criado localmente para armazenar as informações de autenticação. Este arquivo permite que o aplicativo acesse o Google Drive sem precisar solicitar a autorização a cada execução. **Este arquivo também não deve ser incluído no repositório Git, pois contém informações específicas da sua conta.**

5.  **Testando com seus próprios arquivos:**
    * Para testar o aplicativo, crie uma pasta no seu Google Drive que contenha os tipos de arquivos que o projeto se destina a processar (PDF, Excel, PPT, DOC, etc.).
    * Execute o aplicativo. Ele deverá solicitar sua autorização na primeira vez e, em seguida, começar a processar os arquivos na sua conta do Google Drive.

**Importante:**

* **Para testar com seus próprios arquivos, você precisará modificar o código para apontar para a pasta correta no seu Google Drive.** O projeto atualmente está configurado para processar arquivos de uma pasta específica.
* **O projeto foi configurado para utilizar a API do Gemini para gerar embeddings.** Você precisará configurar as credenciais da API do Gemini conforme descrito abaixo.


* **Não compartilhe seu arquivo `credentials.json` com outras pessoas e não o inclua no repositório Git.** Ele contém informações confidenciais da sua conta do Google.
* Cada pessoa que for testar o aplicativo precisará seguir esses passos para obter suas próprias credenciais.


## Configuração da API do Gemini

Este projeto utiliza a API do Gemini para gerar embeddings. Para utilizá-la, você precisará de uma chave de API. Siga os passos abaixo para obter sua chave e configurar o projeto:

1. **Obter uma chave de API do Gemini:**
   * Acesse o [Google AI Studio](https://aistudio.google.com/app/apikey).
   * Faça login com sua conta Google.
   * Clique em "Create API key in new project" ou selecione um projeto existente.
   * Copie a chave de API gerada.

2. **Configurar a chave de API no projeto:**
   * Crie um arquivo chamado `.env` na raiz do projeto (no mesmo diretório que este README.md).
   * Adicione a seguinte linha ao arquivo `.env`, substituindo `SUA_CHAVE_DE_API` pela chave que você copiou:

     ```
     GEMINI_API_KEY=SUA_CHAVE_DE_API
     ```

   * **Importante:** O arquivo `.env` não deve ser incluído no repositório Git, pois contém informações sensíveis. Certifique-se de adicioná-lo ao seu arquivo `.gitignore`.

3. **Instalar a biblioteca do Gemini:**
   * Certifique-se de que a biblioteca `google-generativeai` esteja instalada no seu ambiente virtual. Se não estiver, instale-a usando o pip:
     ```bash
     pip install google-generativeai
     ```

