"""
Define as tarefas que cada agente executará e orquestra o fluxo de trabalho.
"""
from typing import List, Optional

from crewai import Task
from crewai.tools import BaseTool

from Agents import (
    DocumentAnalysisAgent,
    DataMiningAgent,
    ReportingAgent,
    FileManagementAgent,
    ExtractTextTool,
    CountWordsTool,
    FileSummaryTool,
    ListFilesTool,
    CopyFileTool
)
from DocumentTools import MoveFileTool
from ReportGeneretor import GenerateReportTool


def create_document_analysis_tasks(file_path: str) -> List[Task]:
    # Instancie as ferramentas
    extract_text_tool = ExtractTextTool(
        name="extract_text_from_file",
        description="Extracts text content from a file. Supports .pdf, .docx, .txt, .xlsx, .pptx, .doc"
    )
    count_words_tool = CountWordsTool(
        name="count_words",
        description="Counts the number of words in a given text."
    )
    file_summary_tool = FileSummaryTool(
        name="summarize_file_content",
        description="Provides a brief summary of the content of a file."
    )

    # Instancie o agente com as ferramentas (instâncias)
    document_analyzer = DocumentAnalysisAgent()

    # Task auxiliar para encapsular o file_path
    file_path_context_task = Task(
        description="Contexto: Caminho do arquivo para análise.",
        agent=None,
        tools=[],  # Lista vazia de ferramentas (para Task de contexto)
        context=[Task(description="Dados de contexto", agent=None, tools=[], expected_output="O caminho do arquivo.")],
        expected_output="O caminho do arquivo."
    )

    return [
        file_path_context_task,
        Task(
            description=f"Extrair o conteúdo completo do documento localizado em '{file_path}'.",
            agent=document_analyzer,
            tools=[extract_text_tool],  # Passa a INSTÂNCIA da ferramenta para a Task
            context=[file_path_context_task],
            expected_output="O conteúdo completo do arquivo em formato de texto."
        ),
        Task(
            description=f"Contar o número total de palavras no texto extraído de '{file_path}'.",
            agent=document_analyzer,
            tools=[count_words_tool],  # Passa a INSTÂNCIA da ferramenta para a Task
            context=[file_path_context_task],
            expected_output="O número total de palavras no texto extraído."
        ),
        Task(
            description=f"Fornecer um resumo conciso do documento localizado em '{file_path}'. Almejar um resumo de no máximo 500 palavras.",
            agent=document_analyzer,
            tools=[file_summary_tool],  # Passa a INSTÂNCIA da ferramenta para a Task
            context=[file_path_context_task],
            expected_output="Um resumo conciso do conteúdo do arquivo com no máximo 500 palavras."
        )
    ]


def create_data_mining_tasks() -> List[Task]:
    # Instancie as ferramentas
    count_words_tool = CountWordsTool(
        name="count_words",
        description="Counts the number of words in a given text."
    )
    # Instancie o agente com as ferramentas (instâncias)
    data_mining_agent = DataMiningAgent()
    return [
        Task(
            description="Analisar as contagens de palavras dos documentos para identificar tendências ou anomalias.",
            agent=data_mining_agent,
            tools=[count_words_tool],  # Passa a INSTÂNCIA da ferramenta para a Task
            expected_output="Análise de tendências e anomalias nas contagens de palavras."
        )
        # Adicionar mais tarefas de mineração de dados conforme necessário
    ]


def create_reporting_tasks(report_directory: str = "reports") -> List[Task]:
    # Instancie as ferramentas
    generate_report_tool = GenerateReportTool(
        name = "generate_report",
    description = "Generates a report summarizing the key findings from the document analysis."
    )
    list_files_tool = ListFilesTool(
        name="list_files_in_directory",
        description="Lists all files in a given directory."
    )
    copy_file_tool = CopyFileTool(
        name="copy_file",
        description="Copies a file from source to destination."
    )

    # Instancie o agente com as ferramentas (instâncias)
    reporting_agent = ReportingAgent()

    # Task auxiliar para encapsular o report_directory
    report_directory_context_task = Task(
        description="Contexto: Diretório para salvar os relatórios.",
        agent=None,
        tools=[],
        context=[Task(description="Dados de contexto", agent=None, tools=[], expected_output="O diretório para salvar os relatórios.")],
        expected_output="O diretório para salvar os relatórios."
    )

    # Determine the type of generate_report_tool and handle it appropriately
    generate_report_tools: List[Optional[BaseTool]] = []
    if callable(generate_report_tool):
        # If it's a function, it's not a BaseTool instance, so we don't add it directly
        pass  # Or handle it differently if needed
    elif isinstance(generate_report_tool, BaseTool):
        generate_report_tools.append(generate_report_tool)

    return [
        report_directory_context_task,
        Task(
            description=f"Gerar um relatório resumindo as principais descobertas da análise de documentos e mineração de dados. Salvar o relatório no diretório '{report_directory}'.",
            agent=reporting_agent,
            tools=generate_report_tools + [list_files_tool, copy_file_tool],  # Passa as INSTÂNCIAS das ferramentas
            context=[report_directory_context_task],
            expected_output="Relatório gerado no diretório especificado."
        ),
        Task(
            description=f"Listar todos os arquivos no diretório '{report_directory}' para garantir que o relatório foi salvo corretamente.",
            agent=reporting_agent,
            tools=[list_files_tool],  # Passa a INSTÂNCIA da ferramenta
            context=[report_directory_context_task],
            expected_output="Lista de arquivos no diretório do relatório."
        ),
        Task(
            description=f"Copiar o relatório gerado para um local de backup.",
            agent=reporting_agent,
            tools=[copy_file_tool],  # Passa a INSTÂNCIA da ferramenta
            context=[report_directory_context_task],
            expected_output="Relatório copiado para o local de backup."
        )
        # Adicionar mais tarefas de relatório conforme necessário
    ]


def create_file_management_tasks(
        source_directory: str, destination_directory: str
) -> List[Task]:
    # Instancie as ferramentas
    list_files_tool = ListFilesTool(
        name="list_files_in_directory",
        description="Lists all files in a given directory."
    )
    move_file_tool = MoveFileTool(
        name="move_file",
        description="Moves a file from source to destination."
    )

    # Instancie o agente com as ferramentas (instâncias)
    file_management_agent = FileManagementAgent()

    # Tasks auxiliares para os diretórios de origem e destino
    source_directory_context_task = Task(
        description="Contexto: Diretório de origem dos arquivos.",
        agent=None,
        tools=[],
        context=[Task(description="Dados de contexto", agent=None, tools=[], expected_output="O diretório de origem dos arquivos.")],
        expected_output="O diretório de origem dos arquivos."
    )

    destination_directory_context_task = Task(
        description="Contexto: Diretório de destino para mover os arquivos.",
        agent=None,
        tools=[],
        context=[Task(description="Dados de contexto", agent=None, tools=[], expected_output="O diretório de destino para mover os arquivos.")],
        expected_output="O diretório de destino para mover os arquivos."
    )

    return [
        source_directory_context_task,
        destination_directory_context_task,
        Task(
            description=f"Listar todos os arquivos no diretório de origem: '{source_directory}'.",
            agent=file_management_agent,
            tools=[list_files_tool],
            expected_output="Lista de arquivos no diretório de origem."
        ),
        Task(
            description=f"Mover todos os arquivos .txt de '{source_directory}' para '{destination_directory}'.",
            agent=file_management_agent,
            tools=[move_file_tool],
            context=[source_directory_context_task, destination_directory_context_task],
            expected_output="Arquivos movidos do diretório de origem para o diretório de destino."
        )
        # Adicionar mais tarefas de gerenciamento de arquivos conforme necessário
    ]