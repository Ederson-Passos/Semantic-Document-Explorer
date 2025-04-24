import asyncio
from crewai import Crew, Process, Task
from Agents import WebScrapingAgent, AdvancedWebResearchAgent, ReportingAgent
from ReportGeneretor import GenerateReportTool
from WebTools import CrawlAndScrapeSiteTool, ExtractPageStructureTool
import datetime
import os

SITE_URL = "https://www.loc.gov/"
REPORT_DIR = "web_reports"
os.makedirs(REPORT_DIR, exist_ok=True)

async def main():
    web_scraper_agent = WebScrapingAgent()
    advanced_research_agent = AdvancedWebResearchAgent()
    reporting_agent = ReportingAgent()

    crew = Crew(
        agents=[web_scraper_agent, advanced_research_agent, reporting_agent],
        tasks=[
            # Tarefa 1: rastrear o site e extrair informações básicas
            Task(
                agent=advanced_research_agent,
                description=f"Crawl the website {SITE_URL} and extract all the content.",
                tools=CrawlAndScrapeSiteTool(name="crawl_and_scrape")
                # input={"base_url": SITE_URL}
            ),
            # Tarefa 2: analisar a estrutura do site.
            Task(
                agent=advanced_research_agent,
                description="Analyze the structure of the website.",
                tools=ExtractPageStructureTool(
                    name="extract_structure",
                    description="Extracts the structure of a web page."
                )
                # input={"url": SITE_URL}
            ),
            # Tarefa 3: gerar um relatório detalhado.
            Task(
                agent=reporting_agent,
                description="Generate a detailed report on the website's content and structure.",
                tools=GenerateReportTool(
                    name="generate_report",
                    description="Generates a report summarizing the key findings from the document analysis."
                )
                # input={"data":  # Aqui você precisará passar os dados coletados nas tarefas anteriores}
            ),
        ],
        process=Process.sequential
    )

    report = crew.kickoff()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(REPORT_DIR, f"web_report_{timestamp}.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Relatório salvo em: {report_file}")

if __name__ == "__main__":
    asyncio.run(main())