# Requisitos

Certifique-se de ter o Python instalado na versão **3.10.x**. Outras versões podem apresentar problemas de compatibilidade com as dependências do projeto. Você pode baixar o Python 3.10 em [[https://www.python.org/downloads/release/python-3100/](https://www.python.org/downloads/release/python-3100/)].

## Configuração das Credenciais do Google Drive para Testes

Este aplicativo utiliza a API do Google Drive para acessar arquivos para análise. Para testá-lo com sua própria conta do Google Drive, você precisará realizar os seguintes passos para obter suas próprias credenciais e configurar os arquivos necessários:

1.  **Ativar a API do Google Drive no Google Cloud Platform:**
    * Acesse o [Google Cloud Console](https://console.cloud.google.com/).
    * Selecione ou crie um novo projeto.
    * No menu de navegação, vá em "APIs e serviços" > "Biblioteca".
    * Procure por "Google Drive API" e clique em "Ativar".

2.  **Criar Credenciais OAuth 2.0:**
    * No menu de navegação, vá em "APIs e serviços" > "Credenciais".
    * Clique em "Criar credenciais" e selecione "ID do cliente OAuth".
    * Se você ainda não configurou a tela de consentimento OAuth, será solicitado a fazê-lo:
        * Clique em "Configurar tela de consentimento".
        * Selecione o tipo de usuário ("Externo" para testes com contas fora da sua organização, a menos que seja um aplicativo interno).
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

* **Não compartilhe seu arquivo `credentials.json` com outras pessoas e não o inclua no repositório Git.** Ele contém informações confidenciais da sua conta do Google.
* Cada pessoa que for testar o aplicativo precisará seguir esses passos para obter suas próprias credenciais.

## Configurando o Interpretador Python na IntelliJ IDEA / PyCharm

Para garantir que a IDE utilize as dependências corretas do projeto, siga estes passos para configurar o interpretador Python:

1.  Abra o projeto "SemanticDocumentExplorer" na IntelliJ IDEA ou PyCharm.
2.  Vá em **File** > **Settings** (ou **IntelliJ IDEA** > **Preferences** no macOS).
3.  No painel esquerdo, procure por **Project:** (Seu projeto) > **Python Interpreter**.
4.  Você deverá ver uma lista de interpretadores. Se o interpretador do seu `venv` não estiver selecionado:
    * Clique no ícone de engrenagem (⚙️) no canto superior direito da lista de interpretadores.
    * Selecione **Add...**.
    * Na janela que aparecer, selecione **Virtual Environment**.
    * Em "Base interpreter", você pode já ter o Python 3.10 configurado. Em "Location", navegue até a pasta `venv` do seu projeto e selecione o executável do Python:
        * **Windows:** `venv\Scripts\python.exe`
        * **macOS/Linux:** `venv/bin/python`
    * Clique em **OK**.
    * Certifique-se de que o interpretador recém-adicionado do `venv` esteja selecionado na lista de interpretadores do projeto.
5.  Clique em **Apply** e depois em **OK** para salvar as configurações.

Agora, a IDE deverá reconhecer as bibliotecas instaladas no seu ambiente virtual.

## Configurando o Interpretador Python no Visual Studio Code (VS Code)

O VS Code geralmente detecta ambientes virtuais automaticamente, mas se precisar configurar manualmente:

1.  Abra a pasta do seu projeto "SemanticDocumentExplorer" no VS Code.
2.  Abra a paleta de comandos pressionando `Ctrl+Shift+P` (ou `Cmd+Shift+P` no macOS).
3.  Digite "Python: Select Interpreter" e pressione Enter.
4.  Uma lista de interpretadores Python será exibida. Procure pelo interpretador que está dentro da pasta `venv` do seu projeto:
    * O caminho deverá ser algo como:
        * `G:\Meu Drive\Work\Semantic Document Explorer\SemanticDocumentExplorer\venv\Scripts\python.exe` (Windows)
        * `G:\Meu Drive\Work\Semantic Document Explorer\SemanticDocumentExplorer\venv/bin/python` (macOS/Linux)
5.  Selecione o interpretador correto do `venv`.

O VS Code agora usará este interpretador para o seu projeto. Você pode verificar o interpretador selecionado no canto inferior esquerdo da janela do VS Code.