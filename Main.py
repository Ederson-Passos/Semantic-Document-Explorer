import asyncio
import os
import datetime
from crewai import Crew, Process, Task
from Agents import DocumentAnalysisAgent, ReportingAgent
from DocumentTools import ExtractTextTool, CountWordsTool, FileSummaryTool
from ReportGeneretor import GenerateReportTool
from Authentication import GoogleDriveAPI
from DataBaseManager import DataBaseManager


DRIVE_FOLDER_ID = "1lXQ7R5z8NGV1YGUncVDHntiOFX35r6WO"
REPORT_DIR = "google_drive_reports"
TEMP_DIR = "temp_drive_files"


os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)


async def main():
    # Initialize Google Drive API and DataBaseManager
    drive_api = GoogleDriveAPI()
    drive_service = drive_api.service
    db_manager = DataBaseManager(drive_service)

    # Initialize agents and tools
    document_agent = DocumentAnalysisAgent()
    reporting_agent = ReportingAgent()
    extract_text_tool = ExtractTextTool()
    count_words_tool = CountWordsTool()
    summarize_tool = FileSummaryTool()
    generate_report_tool = GenerateReportTool()

    # Get list of files from Google Drive
    files = db_manager.list_files_recursively(DRIVE_FOLDER_ID)

    if not files:
        print("No files found in the specified Google Drive folder.")
        return

    # Download files and create tasks
    tasks = []
    downloaded_files = []  # Keep track of downloaded file paths
    for file in files:
        file_path = os.path.join(TEMP_DIR, file["name"])
        if db_manager.download_file(file["id"], file["name"], TEMP_DIR):
            downloaded_files.append(file_path)
            tasks.append(
                Task(
                    agent=document_agent,
                    description=f"Extract text from the file: {file['name']}",
                    tool=extract_text_tool,
                    input={"file_path": file_path},
                    expected_output=f"Text content of the file: {file['name']}",
                )
            )
        else:
            print(f"Failed to download file: {file['name']}")

    # Add the reporting task
    tasks.append(
        Task(
            agent=reporting_agent,
            description="Generate a detailed report on the content of the files.",
            tool=generate_report_tool,
            input={"data": "Data from previous tasks"},
            expected_output="Complete report on file analysis.",
        )
    )

    # Create and run the crew
    crew = Crew(agents=[document_agent, reporting_agent], tasks=tasks, process=Process.sequential)
    report = crew.kickoff()

    # Save the report
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(REPORT_DIR, f"drive_report_{timestamp}.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Relatório salvo em: {report_file}")

    # Clean up temporary files
    for file_path in downloaded_files:
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error deleting temporary file {file_path}: {e}")
    try:
        os.rmdir(TEMP_DIR)
    except Exception as e:
        print(f"Error removing temporary directory {TEMP_DIR}: {e}")


if __name__ == "__main__":
    asyncio.run(main())