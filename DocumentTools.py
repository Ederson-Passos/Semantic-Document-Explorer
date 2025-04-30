"""Contém as ferramentas que os agentes usarão para interagir com os arquivos."""

from crewai.tools import BaseTool
from TextExtractor import extract_text


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
