"""Contém as ferramentas que os agentes usarão para interagir com a web."""
import time
from crewai.tools import BaseTool
from bs4 import BeautifulSoup
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec

class ScrapeWebsiteTool(BaseTool):
    """
    Extrai o conteúdo textual da url.
    """
    name = "scrape_website_content"
    description = "Scrapes the text content from a given URL. Use this to extract information directly from websites."

    def _run(self, url: str) -> str:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            text_parts = soup.find_all(string=True)
            text_content = '\n'.join(part.strip() for part in text_parts if part.strip())
            return text_content
        except requests.exceptions.RequestException as e:
            return f"Error scraping website: {e}"

class SeleniumScrapingTool(BaseTool):
    """
    Extrai o conteúdo de um site com conteúdo dinâmico usando Selenium.
    """
    name = "scrape_website_with_selenium"
    description = "Scrapes content from a website using Selenium, allowing with JavaScript-rendered content."

    def _run(self, url: str, element_to_wait_for: str = None, wait_timeout: int = 10) -> str:
        driver = None
        try:
            service = ChromeService(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service)
            driver.get(url)

            if element_to_wait_for:
                wait = WebDriverWait(driver, wait_timeout)
                wait.until(ec.presence_of_element_located((By.CSS_SELECTOR, element_to_wait_for)))

            soup = BeautifulSoup(driver.page_source, 'html.parser')
            text_parts = soup.find_all(string=True)
            text_content = '\n'.join(part.strip() for part in text_parts if part.strip())
            driver.quit()
            return text_content
        except Exception as e:
            if 'driver' in locals():
                driver.quit()
            return f"Error scraping website with Selenium: {e}"

class ExtractLinksToll(BaseTool):
    """
    Extrai todos os links de uma página, mapeando a mesma.
    """
    name = "extract_links"
    description = "Extract all links from a given web page."

    def _run(self, url: str) -> list[str]:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            links = [a.get('href') for a in soup.find_all('a', href=True)]
            return links
        except requests.exceptions.RequestException as e:
            return [f"Error extracting links: {e}"]

class ExtractPageStructureTool(BaseTool):
    """
    Busca obter a estrutura da página para análise de conteúdo.
    """
    name = "extract_page_structure"
    description = "Extracts the structure of a web page (headings, paragraphs, lists, etc.)."

    def _run(self, url: str) -> str:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            # Incluir mais extrações.
            structure = ""
            for tag in soup.find_all(['h1', 'h2', 'h3', 'p', 'li']):
                structure += f"<{tag.name}> {tag.text.strip()} </{tag.name}>\n"
            return structure
        except requests.exceptions.RequestException as e:
            return f"Error extracting page structure: {e}"

class ClickAndScrapeTool(BaseTool):
    """
    Usa Selenium para simular clicks e extrair o conteúdo resultante.
    Essencial para páginas com conteúdo dinâmico.
    """
    name = "click_and_scrape"
    description = "Simulates a click on a specific element and the scrapes the resulting content. Requires Selenium."

    def _run(self, url: str, element_selector: str, wait_time: int = 5) -> str:
        driver = None
        try:
            service = ChromeService(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service)
            driver.get(url)
            wait = WebDriverWait(driver, wait_time)
            element = wait.until(ec.presence_of_element_located((By.CSS_SELECTOR, element_selector)))
            element.click()
            time.sleep(wait_time)  # Aguarda o carregamento do conteúdo dinâmico.
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            text_parts = soup.find_all(string=True)
            text_content = '\n'.join(part.strip() for part in text_parts if part.strip())
            return text_content
        except Exception as e:
            return f"Error clicking and scraping: {e}"
        finally:
            if driver:
                driver.quit()

class SimulateMouseMovementTool(BaseTool):
    """
    Simula o movimento do mouse para um elemento específico da página web.
    """
    name = "simulate_mouse_movement"
    description = "Simulates mouse movement to a specific element on the page. Requires Selenium."

    def _run(self, url: str, element_selector: str, wait_time: int = 2) -> str:
        driver = None
        try:
            service = ChromeService(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service)
            driver.get(url)
            wait = WebDriverWait(driver, wait_time)
            element = wait.until(ec.presence_of_element_located((By.CSS_SELECTOR, element_selector)))
            action = ActionChains(driver)
            action.move_to_element(element).perform()
            time.sleep(wait_time)  # Aguarda um breve período após o movimento.
            return f"Mouse moved to element: {element_selector}"
        except Exception as e:
            return f"Error simulating mouse movement: {e}"
        finally:
            if driver:
                driver.quit()

class SimulateScrollTool(BaseTool):
    """
    Simula a rolagem por uma página web.
    """
    name = "simulate_scroll"
    description = "Simulates scrolling on a web page. Requires Selenium."

    def _run(self, url: str, distance: int, wait_time: int = 2) -> str:
        driver = None
        try:
            service = ChromeService(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service)
            driver.get(url)
            driver.execute_script(f"window.scrollBy(0, {distance});")
            time.sleep(wait_time)
            return f"Scrolled by: {distance} pixels"
        except Exception as e:
            return f"Error simulating scroll: {e}"
        finally:
            if driver:
                driver.quit()

class GetElementAttributesTool(BaseTool):
    """
    Obtém os atributos de um elemento específico da página web.
    """
    name = "get_element_attributes"
    description = "Gets attributes of a specific element on the page. Requires Selenium."

    def _run(self, url: str, element_selector: str, attribute: str, wait_time: int = 2) -> str:
        driver = None
        try:
            service = ChromeService(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service)
            driver.get(url)
            wait = WebDriverWait(driver, wait_time)
            element = wait.until(ec.presence_of_element_located((By.CSS_SELECTOR, element_selector)))
            value = element.get_attribute(attribute)
            return f"Attribute '{attribute}' of element '{element_selector}': {value}"
        except Exception as e:
            return f"Error getting element attribute: {e}"
        finally:
            if driver:
                driver.quit()

class SendToGoogleAnalyticsTool(BaseTool):
    """
    Envia os dados para o Google Analytics.
    """
    name = "send_to_google_analytics"
    description = "Sends data to Google Analytics. Requires a Measurement ID and API Secret."

    def _run(self, measurement_id: str, api_secret: str, client_id: str, event_name: str, event_params: dict) -> str:
        url = f"https://www.google-analytics.com/mp/collect?measurement_id={measurement_id}&api_secret={api_secret}"
        payload = {
            "client_id": client_id,
            "events": [{
                "name": event_name,
                "params": event_params
            }]
        }
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return f"Data sent to Google Analytics: {response.status_code}"
        except requests.exceptions.RequestException as e:
            return f"Error sending data to Google Analytics: {e}"