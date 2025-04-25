import asyncio
from crewai import Crew, Process, Task
from Agents import WebScrapingAgent, AdvancedWebResearchAgent, ReportingAgent
from WebTools import CrawlAndScrapeSiteTool, ExtractPageStructureTool
from ReportGeneretor import GenerateReportTool
import datetime
import os

SITE_URL = "https://www.loc.gov/"
REPORT_DIR = "web_reports"
os.makedirs(REPORT_DIR, exist_ok=True)

async def main():
    # Inicializa os agentes
    web_scraper_agent = WebScrapingAgent()
    advanced_research_agent = AdvancedWebResearchAgent()
    reporting_agent = ReportingAgent()

    # Inicializa as Tools
    crawl_tool = CrawlAndScrapeSiteTool(name="optimized_crawl")
    extract_structure_tool = ExtractPageStructureTool(name="extract_structure")
    generate_report_tool = GenerateReportTool()

    # Cria a Crew com os agentes e suas tarefas
    crew = Crew(
        agents=[web_scraper_agent, advanced_research_agent, reporting_agent],
        tasks=[
            # Tarefa 1: Rastrear o site e extrair informações básicas
            Task(
                agent=advanced_research_agent,
                description=f"Crawl the website {SITE_URL} and extract all the content.",
                tool=crawl_tool,
                input={"base_url": SITE_URL, "extract": "text_and_links"},
                expected_output="Comprehensive content and links of the website."
            ),
            # Tarefa 2: Analisar a estrutura do site
            Task(
                agent=advanced_research_agent,
                description="Analyze the structure of the website.",
                tool=extract_structure_tool,
                input={"url": SITE_URL},
                expected_output="Detailed structure of the website."
            ),
            # Tarefa 3: Gerar um relatório detalhado
            Task(
                agent=reporting_agent,
                description="Generate a detailed report on the website's content and structure.",
                tool=generate_report_tool,
                input={"data": "Data from previous tasks"},
                expected_output="Complete report on website analysis."
            ),
        ],
        process=Process.sequential  # As tarefas serão executadas em ordem
    )

    # Executa a Crew e obtém o relatório
    report = crew.kickoff()

    # Salva o relatório em um arquivo
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(REPORT_DIR, f"web_report_{timestamp}.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Relatório salvo em: {report_file}")

if __name__ == "__main__":
    asyncio.run(main())