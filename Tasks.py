"""
Define as tarefas que cada agente executará e orquestra o fluxo de trabalho.
"""
from typing import List, Any

from crewai import Task

from Agents import (
    DocumentAnalysisAgent,
    DataMiningAgent,
    ReportingAgent
)


def create_document_analysis_tasks(file_path: str, llm: Any) -> List[Task]:
    """
    Cria tarefas para extrair, contar palavras e resumir um documento usando o DocumentAnalysisAgent.
    Args:
        file_path (str): O caminho para o arquivo a ser analisado.
        llm (Any): A instância do LLM a ser usada pelo agente.
    Returns:
        List[Task]: Uma lista contendo a tarefa de análise do documento.
    """
    # Instancie o agente passando o LLM
    document_analyzer = DocumentAnalysisAgent(llm=llm)

    # Tarefa única combinando extração, contagem e resumo.
    analysis_task = Task(
        description=(
            f"1. Use a ferramenta 'extract_text_from_file' para extrair o conteúdo completo "
            f"do documento em: '{file_path}'.\n"
            f"2. Use a ferramenta 'count_words' para contar o número total de palavras no texto extraído.\n"
            f"3. Use suas capacidades de raciocínio (LLM) para gerar um resumo conciso "
            f"(1-2 parágrafos, máx 500 palavras) da mensagem principal do documento.\n"
            "Certifique-se de que o resultado final seja um dicionário Python estruturado."
        ),
        agent=document_analyzer,
        expected_output=(
            "Um dicionário Python contendo:\n"
            "- 'file_path': O caminho original do arquivo analisado (string).\n"
            "- 'full_text': O conteúdo completo do texto extraído (string).\n"
            "- 'word_count': A contagem total de palavras (integer).\n"
            "- 'summary': Um resumo conciso do documento gerado por você (string)."
        )
    )
    return [analysis_task]


def create_data_mining_tasks(llm: Any) -> List[Task]:
    """
    Cria tarefas para analisar os resultados das análises de documentos (passados como contexto).
    Args:
        llm (Any): A instância do LLM a ser usada pelo agente.
    Returns:
        List[Task]: Uma lista contendo a tarefa de mineração de dados.
    """
    # Instancie o agente passando o LLM
    data_mining_agent = DataMiningAgent(llm=llm)

    # Tarefa para analisar resultados agregados (passados via contexto pela Crew)
    mining_task = Task(
        description=(
            "Analise os resultados coletados (contagens de palavras, resumos) das tarefas anteriores de "
            "análise de documentos (fornecidos como contexto). "
            "Identifique quaisquer tendências, padrões, anomalias ou métricas chave notáveis entre os documentos. "
            "Concentre-se em comparar contagens de palavras e resumir temas comuns ou discrepâncias "
            "encontradas nos resumos. "
            "Use a ferramenta 'count_words' se precisar recontar ou verificar partes específicas "
            "do texto no contexto."
        ),
        agent=data_mining_agent,
        expected_output=(
            "Um relatório de análise textual detalhando:\n"
            "- Tendências ou padrões observados nas contagens de palavras entre documentos.\n"
            "- Anomalias ou outliers nas contagens de palavras.\n"
            "- Temas comuns ou descobertas chave sintetizadas a partir dos resumos dos documentos.\n"
            "- Quaisquer outras métricas ou insights relevantes derivados dos dados combinados."
        )
    )
    return [mining_task]


def create_reporting_tasks(llm: Any, report_directory: str = "reports") -> List[Task]:
    """
    Cria tarefas para consolidar resultados e instruir o ReportingAgent a salvar um relatório.

    Args:
        llm (Any): A instância do LLM a ser usada pelo agente.
        report_directory (str): O diretório onde o relatório deve ser salvo.

    Returns:
        List[Task]: Uma lista contendo a tarefa de geração de relatório.
    """
    # Instancie o agente passando o LLM
    reporting_agent = ReportingAgent(llm=llm)

    # Tarefa para gerar e salvar o relatório consolidado
    generate_save_report_task = Task(
        description=(
            f"Consolide TODAS as descobertas das tarefas anteriores de análise de documentos e mineração "
            f"de dados (fornecidas como contexto). "
            f"Estruture essas descobertas em um relatório final coeso. O relatório deve incluir as informações "
            f"de cada arquivo (contagem de palavras, resumo) e a análise de mineração de dados.\n"
            f"Após consolidar o conteúdo do relatório, use a ferramenta 'save_analysis_report' para salvar este "
            f"relatório consolidado como um arquivo de texto. "
            f"O input para a ferramenta 'save_analysis_report' deve ser um dicionário Python contendo os "
            f"resultados consolidados (você pode precisar estruturá-lo adequadamente, por exemplo, com chaves "
            f"sendo nomes de arquivos ou 'mining_analysis'). "
            f"Certifique-se de que o diretório para salvar seja '{report_directory}'."
        ),
        agent=reporting_agent,
        expected_output=(
            f"A confirmação de que o relatório final consolidado foi gerado com sucesso e salvo como "
            f"um arquivo de texto "
            f"usando a ferramenta 'save_analysis_report' no diretório '{report_directory}'. "
            f"Inclua na sua resposta final o caminho completo do arquivo salvo retornado pela ferramenta."
        )
    )
    return [generate_save_report_task]