"""
Contém a lógica para gerar relatórios.
"""
import os
from crewai.tools import BaseTool

class GenerateReportTool(BaseTool):
    name: str = "generate_report"
    description: str = "Generates a report summarizing the key findings from the document analysis."

    def _run(self, analysis_results:  dict, report_directory: str = "reports") -> str:
        """
        Generates a report based on the analysis results.
        Args:
            analysis_results (dict): a dictionary containing the results of the document analysis.
            report_directory (str): the dictionary where the report should be saved.
        Returns:
            str: the path to the generated report.
        """

        os.makedirs(report_directory, exist_ok=True)
        report_path = os.path.join(report_directory, "report.txt")
        with open(report_path, "w") as f:
            f.write("Document Analysis Report\n")
            f.write("-----------------------\n\n")
            for file_path, results in analysis_results.items():
                f.write(f"File: {os.path.basename(file_path)}\n")
                f.write(f"  Word Count: {results.get('word_count', 'N/A')}\n")
                f.write(f"  Summary: {results.get('summary', 'N/A')}\n")
                f.write("\n")

                # Adicionar um relatório mais sofisticado (visualização de dados, análise de tendência)

        return report_path