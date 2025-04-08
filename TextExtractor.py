import PyPDF2
from docx import Document
from openpyxl import load_workbook
from pptx import Presentation


def extract_text_from_pdf(file_path):
    """Extrai texto de um arquivo PDF."""
    text = ""
    try:
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
    except Exception as e:
        print(f"Erro ao extrair texto em PDF: {e}")

    return text


def extract_text_from_docx(file_path):
    """Extrai texto de um arquivo DOCX."""
    text = ""
    try:
        doc = Document(file_path)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        print(f"Erro ao extrair texto do DOCX: {e}")

    return text


def extract_text_from_txt(file_path):
    """Extrai texto de um arquivo TXT."""
    text = ""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
    except Exception as e:
        print(f"Erro ao extrair texto do TXT: {e}")

    return text


def extract_text_from_xlsx(file_path):
    """Extrai texto de um arquivo XLSX."""
    text = ""
    try:
        workbook = load_workbook(filename=file_path, read_only=True)
        for sheet_name in workbook.sheetnames:
            sheet = workbook[sheet_name]
            for row in sheet.rows:
                row_text = " ".join(str(cell.value) for cell in row if cell.value is not None)
                text += row_text + "\n"
    except Exception as e:
        print(f"Erro ao extrair texto do XLSX: {e}")

    return text


def extract_text_from_ppt(file_path):
    """Extrai texto de um arquivo PPT."""
    text = ""
    try:
        prs = Presentation(file_path)
        for slide in prs.slides:
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for paragraph in shape.text_frame.paragraphs:
                        for run in paragraph.runs:
                            text += run.text + " "
                    text += "\n"
    except Exception as e:
        print(f"Erro ao processar o arquivo PPT: {e}")

    return text


def extractor(files):
    for file in files:
        file_name = file["name"]
        file_path = file_name  # Assumindo que o arquivo foi baixado no mesmo diretório
        if file_name.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        elif file_name.endswith(".docx"):
            text = extract_text_from_docx(file_path)
        elif file_name.endswith(".txt"):
            text = extract_text_from_txt(file_path)
        elif file_name.endswith(".xlsx"):
            text = extract_text_from_xlsx(file_path)
        else:
            text = ""
        if text:
            print(f"Texto extraído de {file_name}:\n{text}\n")

    return text