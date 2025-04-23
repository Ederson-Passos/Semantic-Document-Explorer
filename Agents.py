from crewai import Agent
from DocumentTools import (ExtractTextTool, CountWordsTool, FileSummaryTool, ListFilesTool, CopyFileTool,
                           GetFileSizeTool, CreateDirectoryTool, DeleteDirectoryTool, DeleteFileTool, MoveFileTool)
from WebTools import (ScrapeWebsiteTool, SeleniumScrapingTool, ExtractLinksToll, ExtractPageStructureTool,
                      ClickAndScrapeTool, SimulateMouseMovementTool, SimulateScrollTool, GetElementAttributesTool,
                      SendToGoogleAnalyticsTool, CrawlAndScrapeSiteTool)
from ReportGeneretor import GenerateReportTool

class DocumentAnalysisAgent(Agent):
    """
    Responsável por extrair o texto dos documentos e realizar contagem de palavras e resumos.
    """
    def __init__(self):
        super().__init__(
            role="Document Analyst",
            goal="Extract information from various document types and prepare them for analysis.",
            tools=[
                ExtractTextTool(
                    name="extract_text_from_file",
                    description="Extracts text content from a file. Supports .pdf, .docx, .txt, .xlsx, .pptx, .doc"
                ),
                CountWordsTool(
                    name="count_words",
                    description="Counts the number of words in a given text."
                ),
                FileSummaryTool(
                    name="summarize_file_content",
                    description="Provides a brief summary of the content of a file."
                )
            ],
            memory=True,
            verbose=True
        )

class DataMiningAgent(Agent):
    """
    Encontra padrões e métricas nos dados extraídos.
    """
    def __init__(self):
        super().__init__(
            role="Data Mining Expert",
            goal="Identify patterns, trends, and key metrics within the extracted data.",
            tools=[
                CountWordsTool(
                    name="count_words",
                    description="Counts the number of words in a given text."
                )
            ],
            memory=True,
            verbose=True
        )

class ReportingAgent(Agent):
    """
    Gera os relatórios finais e cria novos documentos.
    """
    def __init__(self):
        super().__init__(
            role="Reporting Specialist",
            goal="Generate comprehensive reports based on the analyzed data and create new documents.",
            tools=[
                GenerateReportTool,
                ListFilesTool(
                    name="list_files_in_directory",
                    description="Lists all files in a given directory."
                ),
                CopyFileTool(
                    name="copy_file",
                    description="Copies a file from source to destination."
                ),
                GetFileSizeTool(
                    name="get_file_size",
                    description="Gets the size of a file in bytes."
                ),
                CreateDirectoryTool(
                    name="create_directory",
                    description="Creates a new directory."
                ),
                DeleteDirectoryTool(
                    name="delete_directory",
                    description="Deletes a directory and its contents."
                ),
                DeleteFileTool(
                    name="delete_file",
                    description="Deletes a file."
                ),
                MoveFileTool(
                    name="move_file",
                    description="Moves a file from source to destination."
                )
            ],
            memory=True,
            verbose=True
        )

class FileManagementAgent(Agent):
    """
    Organiza os arquivos, move, copia, deleta, etc.
    """
    def __init__(self):
        super().__init__(
            role="File Management Specialist",
            goal="Organize, manage, and maintain the document repository.",
            tools=[
                ListFilesTool(
                    name="list_files_in_directory",
                    description="Lists all files in a given directory."
                ),
                CopyFileTool(
                    name="copy_file",
                    description="Copies a file from source to destination."
                ),
                MoveFileTool(
                    name="move_file",
                    description="Moves a file from source to destination."
                ),
                DeleteFileTool(
                    name="delete_file",
                    description="Deletes a file."
                ),
                CreateDirectoryTool(
                    name="create_directory",
                    description="Creates a new directory."
                ),
                DeleteDirectoryTool(
                    name="delete_directory",
                    description="Deletes a directory and its contents."
                ),
                GetFileSizeTool(
                    name="get_file_size",
                    description="Gets the size of a file in bytes."
                )
            ],
            memory=True,
            verbose=True
        )

class WebScrapingAgent(Agent):
    """
    Responsável por extrair informações de páginas web.
    """
    def __init__(self):
        super().__init__(
            role="Web Scraper",
            goal="Extract relevant information from websites efficiently and accurately.",
            tools=[
                ScrapeWebsiteTool(
                    name="scrape_website_content",
                    description="Scrapes the text content from a given URL. Use this to extract information directly "
                                "from websites."
                ),
                SeleniumScrapingTool(
                    name="scrape_website_with_selenium",
                    description="Scrapes content from a website using Selenium, allowing with JavaScript-rendered "
                                "content. Use when regular scraping fails or when dynamic content is needed. Input "
                                "should be the URL and optionally the CSS selector of an element to wait for."
                )
            ],
            memory=True,
            verbose=True
        )

class AdvanceWebScrapingAgent(Agent):
    """
    Especializado em scraping que requer manipulação de JavaScript ou espera por elementos específicos.
    """
    def __init__(self):
        super().__init__(
            role="Advanced Web Scraper",
            goal="Extract complex data from websites that rely heavily on JavaScript or require specific interactions.",
            tools=[
                SeleniumScrapingTool(
                    name="scrape_website_with_selenium",
                    description="Scrapes content from a website using Selenium, allowing with JavaScript-rendered "
                                "content. Use when regular scraping fails or when dynamic content is needed. Input "
                                "should be the URL and the CSS selector of an element to wait for."
                )
            ],
            memory=True,
            verbose=True
        )

class AdvancedWebResearchAgent(Agent):
    """
    Agente para realizar pesquisas web complexas, seguindo links, estruturando
    informações e interagindo com elementos da página.
    """
    def __init__(self):
        super().__init__(
            role="Advanced Web Researcher",
            goal="Conduct in-depth web research, explore multiple pages, and gather structures information.",
            tools=[
                ScrapeWebsiteTool(
                    name="scrape_website_content",
                    description="Scrapes the text content from a given URL."
                ),
                SeleniumScrapingTool(
                    name="scrape_website_with_selenium",
                    description="Scrapes content from a website using Selenium."
                ),
                ExtractLinksToll(
                    name="extract_links",
                    description="Extracts all links from a given web page."
                ),
                ExtractPageStructureTool(
                    name="extract_page_structure",
                    description="Extracts the structure of a web page."
                ),
                ClickAndScrapeTool(
                    name="click_and_scrape",
                    description="Simulates clicks and scrapes resulting content."
                )
            ],
            memory=True,
            verbose=True
        )

class BehaviorTrackingAgent(Agent):
    """
    Agente para rastrear o comportamento do usuário em uma página web,
    incluindo movimentos de mouse, cliques, rolagem e interações com elementos.
    """
    def __init__(self):
        super().__init__(
            role="User Behavior Tracker",
            goal="Monitor and record user interactions on websites to understand behavior patterns.",
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
            verbose=True
        )

class AnalyticsReportingAgent(Agent):
    """
    Agente para enviar dados coletados para ferramentas de analytics,
    como o Google Analytics.
    """
    def __init__(self):
        super().__init__(
            role="Analytics Reporter",
            goal="Send collected data to analytics platforms for tracking and analysis.",
            tools=[
                SendToGoogleAnalyticsTool(
                    name="send_to_google_analytics",
                    description="Sends data to Google Analytics."
                )
            ],
            memory=True,
            verbose=True
        )

class SiteCrawlerAgent(Agent):
    """
    Agente para rastrear um site inteiro, extrair conteúdo e identificar informações relevantes.
    """
    def __init__(self):
        super().__init__(
            role="Website Crawler and Content Extractor",
            goal="Explore a website, extract content from all relevant pages, and identify key information.",
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
            verbose=True
        )