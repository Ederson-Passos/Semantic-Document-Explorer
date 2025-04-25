"""Contém as ferramentas que os agentes usarão para interagir com os arquivos."""
from crewai.tools import BaseTool
from TextExtractor import extract_text
import os
import shutil

class ExtractTextTool(BaseTool):
    name: str = "extract_text_from_file"
    description: str = "Extracts text content from a file. Supports .pdf, .docx, .txt, .xlsx, .pptx, .doc"

    def _run(self, file_path: str) -> str:
        return extract_text(file_path)

class CountWordsTool(BaseTool):
    name: str = "count_words"
    description: str = "Counts the number of words in a given text."

    def _run(self, text: str) -> int:
        return len(text.split())

class FileSummaryTool(BaseTool):
    name: str = "summarize_file_content"
    description: str = "Provides a brief summary of the content of a file."

    def _run(self, file_path: str, max_length: int = 500) -> str:
        text = extract_text(file_path)
        if text:
            return text[:max_length]
        return "No summary available."

class ListFilesTool(BaseTool):
    name: str = "list_files_in_directory"
    description: str = "Lists all files in a given directory."

    def _run(self, directory_path: str) -> list[str]:
        try:
            return os.listdir(directory_path)
        except FileNotFoundError:
            return []

class CopyFileTool(BaseTool):
    name: str = "copy_file"
    description: str = "Copies a file from source to destination."

    def _run(self, source_path: str, destination_path: str) -> str:
        try:
            shutil.copy2(source_path, destination_path)
            return f"File copied from '{source_path}' to '{destination_path}'"
        except FileNotFoundError:
            return f"Error: File not found at '{source_path}"
        except Exception as e:
            return f"Error copying file: {e}"

class GetFileSizeTool(BaseTool):
    name: str = "get_file_size"
    description: str = "Gets the size of a file in bytes."

    def _run(self, file_path: str) -> int:
        try:
            return os.path.getsize(file_path)
        except FileNotFoundError:
            return 0

class DeleteFileTool(BaseTool):
    name: str = "delete_file"
    description: str = "Deletes a file."

    def _run(self, file_path: str) -> str:
        try:
            os.remove(file_path)
            return f"File '{file_path}' deleted successfully."
        except FileNotFoundError:
            return f"Error: File not found at '{file_path}'"
        except Exception as e:
            return f"Error deleting file: {e}"

class MoveFileTool(BaseTool):
    name: str = "move_file"
    description: str = "Moves a file from source to destination."

    def _run(self, source_path: str,  destination_path: str) -> str:
        try:
            shutil.move(source_path, destination_path)
            return f"File moved from '{source_path}' to '{destination_path}'"
        except FileNotFoundError:
            return f"Error: File not found at '{source_path}'"
        except Exception as e:
            return f"Error moving file: {e}"

class CreateDirectoryTool(BaseTool):
    name: str = "create_directory"
    description: str = "Creates a new directory."

    def _run(self, directory_path: str) -> str:
        try:
            os.makedirs(directory_path, exist_ok=True)
            return f"Directory '{directory_path}' created successfully."
        except Exception as e:
            return f"Error creating directory: {e}"

class DeleteDirectoryTool(BaseTool):
    name: str = "delete_directory"
    description: str = "Deletes a directory and its contents."

    def _run(self, directory_path: str) -> str:
        try:
            shutil.rmtree(directory_path)
            return f"Directory '{directory_path}' and its contents deleted successfully."
        except FileNotFoundError:
            return f"Error: Directory not found at '{directory_path}'"
        except Exception as e:
            return f"Error deleting directory: {e}"