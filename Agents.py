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
            backstory="I am a meticulous document analyst, skilled in extracting key information from a wide range of "
                      "document formats. My expertise lies in preparing documents for in-depth analysis.",
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
            backstory="I am a seasoned data mining expert with a keen eye for detail. My mission is to uncover hidden "
                      "patterns and trends within complex datasets, providing valuable insights.",
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
    Agente para gerar relatórios detalhados sobre o conteúdo da web.
    """
    def __init__(self):
        super().__init__(
            role="Web Analysis Reporter",
            goal="Generate comprehensive reports on website content and structure.",
            backstory="I specialize in analyzing web data and generating detailed reports, providing insights into website content, structure, and metrics.",
            tools=[
                GenerateReportTool(),  # Instanciando GenerateReportTool
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
            backstory="I am a highly organized file management specialist, dedicated to maintaining a clean and "
                      "efficient document repository. My skills ensure that all files are properly stored and "
                      "accessible.",
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
            verbose=True
        )

class AdvancedWebScrapingAgent(Agent):
    """
    Especializado em scraping que requer manipulação de JavaScript ou espera por elementos específicos.
    """
    def __init__(self):
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
            verbose=True
        )

class AdvancedWebResearchAgent(Agent):
    """
    Agente para realizar pesquisas web complexas, seguindo links e analisando a estrutura.
    """
    def __init__(self):
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
            backstory="I am an analytics reporter, specializing in sending collected data to analytics platforms. My "
                      "expertise ensures that data is properly tracked and analyzed for valuable insights.",
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
            verbose=True
        )