"""
Define os Agentes especializados para análise de documentos e geração de relatórios.
"""
import datetime
import os
import traceback

from crewai import Agent, LLM
from typing import Any

from crewai.tools import BaseTool

from DocumentTools import ExtractTextTool, CountWordsTool
from WebTools import (ScrapeWebsiteTool, SeleniumScrapingTool, ExtractLinksToll, ExtractPageStructureTool,
                      ClickAndScrapeTool, SimulateMouseMovementTool, SimulateScrollTool, GetElementAttributesTool,
                      SendToGoogleAnalyticsTool, CrawlAndScrapeSiteTool)


class DocumentAnalysisAgent(Agent):
    """
    Responsável por extrair informações concisas dos documentos: contagem de palavras e resumos.
    Utiliza ferramentas para extração e contagem, e seu próprio LLM para sumarização.
    """
    def __init__(self, llm: Any):
        super().__init__(
            role="Concise Document Analyst",
            goal="Extract key information (like word count) from document content using tools, "
                 "and generate a concise summary using your own reasoning capabilities. "
                 "Prepare the structured data for reporting.",
            backstory="I am a meticulous document analyst. I use tools to extract text and count "
                      "words, and then apply my language understanding to summarize the core message efficiently. "
                      "My goal is to provide structured, actionable insights from documents.",
            tools=[
                ExtractTextTool(),
                CountWordsTool(),
            ],
            memory=False,
            verbose=False,
            llm=llm
        )

class DataMiningAgent(Agent):
    """
    Encontra padrões e métricas nos dados extraídos.
    """
    def __init__(self, llm: Any):
        super().__init__(
            role="Data Mining Expert",
            goal="Identify patterns, trends, and key metrics within the extracted data.",
            backstory="I am a seasoned data mining expert with a keen eye for detail. "
                      "My mission is to uncover hidden "
                      "patterns and trends within complex datasets, providing valuable insights.",
            tools=[
                CountWordsTool()
            ],
            memory=True,
            verbose=False,
            llm=llm
        )

class GenerateReportTool(BaseTool):
    name: str = "save_analysis_report"
    description: str = ("Saves the provided analysis results dictionary to a text file. "
                        "Input should be a dictionary where keys are filenames and values are analysis results.")

    def _run(self, analysis_results: dict, report_directory: str = "reports", filename_prefix: str = "analysis") -> str:
        """
        Saves the analysis results dictionary to a file.
        Args:
            analysis_results (dict): Dictionary containing analysis results.
            report_directory (str): Directory to save the report.
            filename_prefix (str): Prefix for the report filename.
        Returns:
            str: Path to the saved report file.
        """
        if not isinstance(analysis_results, dict):
            return "Error: Input must be a dictionary."

        try:
            # Cria o diretório especificado pela variável 'report_directory'.
            # O argumento 'exist_ok=True' garante que nenhum erro será lançado se o diretório já existir.
            os.makedirs(report_directory, exist_ok=True)
            # Obtém a data e hora atuais e formata como uma string no formato "AnoMesDia_HoraMinutoSegundo".
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"{filename_prefix}_report_{timestamp}.txt"  # Cria o nome do arquivo de relatório.
            report_path = os.path.join(report_directory, report_filename)  # Constrói o caminho.

            with open(report_path, "w", encoding="utf-8") as f:
                f.write(f"Document Analysis Report - {timestamp}\n")
                f.write("========================================\n\n")
                for file_key, results in analysis_results.items():
                    f.write(f"File: {str(file_key)}\n")
                    if isinstance(results, dict):
                        # Converte explicitamente para string
                        word_count_str = str(results.get('word_count', 'N/A'))
                        summary_str = str(results.get('summary', 'N/A'))
                        f.write(f"  Word Count: {word_count_str}\n")
                        f.write(f"  Summary: {summary_str}\n")
                    else:
                        # Converte explicitamente para string
                        analysis_result_str = str(results)
                        f.write(f"  Analysis Result: {analysis_result_str}\n")
                    f.write("-" * 40 + "\n\n")
            print(f"Report saved to: {report_path}")
            return report_path
        except Exception as e:
            error_message = f"Error saving report: {e}"
            print(error_message)
            traceback.print_exc()
            return error_message


class ReportingAgent(Agent):
    """
    Agente para consolidar análises e gerar relatórios em Markdown.
    """
    def __init__(self, llm: Any):
        super().__init__(
            role="Document Analysis Consolidator",
            goal="Consolidate analysis results (summaries and word counts) from multiple documents "
                 "(provided as context from previous tasks) and generate a cohesive Markdown report.",
            backstory="I specialize in synthesizing information from multiple document analyses. "
                      "I take the structured results (summaries, word counts) from previous steps and "
                      "organize them into a clear, well-formatted final Markdown report, "
                      "highlighting the key findings from each document.",
            tools=[
                GenerateReportTool()  # Gera o relatório em .txt
            ],
            memory=False,
            verbose=False,
            llm=llm
        )

class WebScrapingAgent(Agent):
    """
    Responsável por extrair informações de páginas web.
    """
    def __init__(self, llm: Any):
        super().__init__(
            role="Web Content Extractor",
            goal="Extract and summarize content from web pages.",
            backstory="I am an expert in extracting information from web pages, identifying key details and "
                      "summarizing content effectively.",
            tools=[
                ScrapeWebsiteTool(
                    name="scrape_website",
                    description="Scrapes text content from a website."
                ),
                ExtractLinksToll(
                    name="extract_links",
                    description="Extracts all links from a web page."
                )
            ],
            memory=True,
            verbose=False,
            llm=llm
        )

class AdvancedWebScrapingAgent(Agent):
    """
    Especializado em scraping que requer manipulação de JavaScript ou espera por elementos específicos.
    """
    def __init__(self, llm: Any):
        super().__init__(
            role="Advanced Web Scraper",
            goal="Extract complex data from websites that rely heavily on JavaScript or require specific interactions.",
            backstory="I am an advanced web scraper, adept at navigating and extracting data from even the most "
                      "complex websites. My expertise lies in handling JavaScript-heavy sites and dynamic content.",
            tools=[
                SeleniumScrapingTool(
                    name="scrape_website_with_selenium",
                    description="Scrapes content from a website using Selenium, allowing with JavaScript-rendered "
                                "content. Use when regular scraping fails or when dynamic content is needed. Input "
                                "should be the URL and the CSS selector of an element to wait for."
                )
            ],
            memory=True,
            verbose=False,
            llm=llm
        )

class AdvancedWebResearchAgent(Agent):
    """
    Agente para realizar pesquisas web complexas, seguindo links e analisando a estrutura.
    """
    def __init__(self, llm: Any):
        super().__init__(
            role="Advanced Web Researcher",
            goal="Explore websites deeply, following links, and structuring information.",
            backstory="I am a highly skilled web researcher, capable of navigating complex websites, following links, "
                      "and organizing information into structured formats.",
            tools=[
                SeleniumScrapingTool(
                    name="scrape_with_selenium",
                    description="Scrapes content using Selenium for dynamic content."
                ),
                ExtractPageStructureTool(
                    name="extract_structure",
                    description="Extracts the structure of a web page."
                ),
                CrawlAndScrapeSiteTool(
                    name="crawl_and_scrape",
                    description="Crawls and scrapes an entire website."
                )
            ],
            memory=True,
            verbose=False,
            llm=llm
        )

class BehaviorTrackingAgent(Agent):
    """
    Agente para rastrear o comportamento do usuário em uma página web,
    incluindo movimentos de mouse, cliques, rolagem e interações com elementos.
    """
    def __init__(self, llm: Any):
        super().__init__(
            role="User Behavior Tracker",
            goal="Monitor and record user interactions on websites to understand behavior patterns.",
            backstory="I am a dedicated user behavior tracker, skilled in monitoring and recording user interactions on"
                      "websites. My goal is to provide insights into user behavior patterns and improve user "
                      "experience.",
            tools=[
                ScrapeWebsiteTool(
                    name="scrape_website_content",
                    description="Scrapes the text content from a given URL."
                ),
                SeleniumScrapingTool(
                    name="scrape_website_with_selenium",
                    description="Scrapes content from a website using Selenium."
                ),
                SimulateMouseMovementTool(
                    name="simulate_mouse_movement",
                    description="Simulates mouse movement to a specific element."
                ),
                SimulateScrollTool(
                    name="simulate_scroll",
                    description="Simulates scrolling on a web page."
                ),
                ClickAndScrapeTool(
                    name="click_and_scrape",
                    description="Simulates clicks and scrapes resulting content."
                ),
                GetElementAttributesTool(
                    name="get_element_attributes",
                    description="Gets attributes of a specific element."
                )
            ],
            memory=True,
            verbose=False,
            llm=llm
        )

class AnalyticsReportingAgent(Agent):
    """
    Agente para enviar dados coletados para ferramentas de analytics,
    como o Google Analytics.
    """
    def __init__(self, llm: Any):
        super().__init__(
            role="Analytics Reporter",
            goal="Send collected data to analytics platforms for tracking and analysis.",
            backstory="I am an analytics reporter, specializing in sending collected data to analytics platforms. My "
                      "expertise ensures that data is properly tracked and analyzed for valuable insights.",
            tools=[
                SendToGoogleAnalyticsTool(
                    name="send_to_google_analytics",
                    description="Sends data to Google Analytics."
                )
            ],
            memory=True,
            verbose=False,
            llm = llm
        )

class SiteCrawlerAgent(Agent):
    """
    Agente para rastrear um site inteiro, extrair conteúdo e identificar informações relevantes.
    """
    def __init__(self, llm: Any):
        super().__init__(
            role="Website Crawler and Content Extractor",
            goal="Explore a website, extract content from all relevant pages, and identify key information.",
            backstory="I am a website crawler and content extractor, skilled in exploring websites and extracting "
                      "content from all relevant pages. My goal is to identify key information and provide "
                      "comprehensive data.",
            tools=[
                CrawlAndScrapeSiteTool(
                    name="crawl_and_scrape_site",
                    description="Crawls a website and scrapes content from each page."
                ),
                ScrapeWebsiteTool(
                    name="scrape_website_content",
                    description="Scrapes the text content from a given URL."
                ),
                ExtractLinksToll(
                    name = "extract_links",
                    description = "Extract all links from a given web page."
                ),
                ExtractPageStructureTool(
                    name="extract_page_structure",
                    description="Extracts the structure of a web page."
                )
            ],
            memory=True,
            verbose=False,
            llm=llm
        )