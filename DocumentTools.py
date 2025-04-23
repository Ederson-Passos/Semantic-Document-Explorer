"""Contém as ferramentas que os agentes usarão para interagir com os arquivos."""

from crewai.tools import BaseTool
from TextExtractor import extract_text, TEMP_DOWNLOAD_FOLDER
import os
import shutil

class ExtractTextTool(BaseTool):
    name = "extract_text_from_file"
    description = "Extracts text content from a file. Supports .pdf, .docx, .txt, .xlsx, .pptx, .doc"

    def _run(self, file_path: str) -> str:
        return extract_text(file_path)

class CountWordsTool(BaseTool):
    name = "count_words"
    description = "Counts the number of words in a given text."

    def _run(self, text: str) -> int:
        return len(text.split())

class FileSummaryTool(BaseTool):
    name = "summarize_file_content"
    description = "Provides a brief summary of the content of a file."

    def _run(self, file_path: str, max_length: int = 500) -> str:
        text = extract_text(file_path)
        if text:
            return text[:max_length]
        return "No summary available."

class ListFilesTool(BaseTool):
    name = "list_files_in_directory"
    description = "Lists all files in a given directory."

    def _run(self, directory_path: str) -> list[str]:
        try:
            return os.listdir(directory_path)
        except FileNotFoundError:
            return []

class CopyFileTool(BaseTool):
    name = "copy_file"
    description = "Copies a file from source to destination."

    def _run(self, source_path: str, destination_path: str) -> str:
        try:
            shutil.copy2(source_path, destination_path)
            return f"File copied from '{source_path}' to '{destination_path}'"
        except FileNotFoundError:
            return f"Error: File not found at '{source_path}"
        except Exception as e:
            return f"Error copying file: {e}"

class GetFileSizeTool(BaseTool):
    name = "get_file_size"
    description = "Gets the size of a file in bytes."

    def _run(self, file_path: str) -> int:
        try:
            return os.path.getsize(file_path)
        except FileNotFoundError:
            return 0

class DeleteFileTool(BaseTool):
    name = "delete_file"
    description = "Deletes a file."

    def _run(self, file_path: str) -> str:
        try:
            os.remove(file_path)
            return f"File '{file_path}' deleted successfully."
        except FileNotFoundError:
            return f"Error: File not found at '{file_path}'"
        except Exception as e:
            return f"Error deleting file: {e}"

class MoveFileTool(BaseTool):
    name = "move_file"
    description = "Moves a file from source to destination."

    def _run(self, source_path: str,  destination_path: str) -> str:
        try:
            shutil.move(source_path, destination_path)
            return f"File moved from '{source_path}' to '{destination_path}'"
        except FileNotFoundError:
            return f"Error: File not found at '{source_path}'"
        except Exception as e:
            return f"Error moving file: {e}"

class CreateDirectoryTool(BaseTool):
    name = "create_directory"
    description = "Creates a new directory."

    def _run(self, directory_path: str) -> str:
        try:
            os.makedirs(directory_path, exist_ok=True)
            return f"Directory '{directory_path}' created successfully."
        except Exception as e:
            return f"Error creating directory: {e}"

class DeleteDirectoryTool(BaseTool):
    name = "delete_directory"
    description = "Deletes a directory and its contents."

    def _run(self, directory_path: str) -> str:
        try:
            shutil.rmtree(directory_path)
            return f"Directory '{directory_path}' and its contents deleted successfully."
        except FileNotFoundError:
            return f"Error: Directory not found at '{directory_path}'"
        except Exception as e:
            return f"Error deleting directory: {e}"