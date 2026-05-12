import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from PyPDF2 import PdfReader
from crewai import Agent, Crew, LLM, Process, Task
from fpdf import FPDF

load_dotenv()

INPUT_DIR = Path("inputs")
OUTPUT_DIR = Path("outputs")
MODEL_NAME = "groq/llama-3.3-70b-versatile"
API_ENV_KEYS = ["CREWAI_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY", "API_KEY"]


def ensure_directories() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def locate_latest_pdf(directory: Path) -> Optional[Path]:
    """находит самый последний PDF-файл"""
    pdf_files = [path for path in directory.glob("*.pdf") if path.is_file()]
    if not pdf_files:
        return None
    return max(pdf_files, key=lambda path: path.stat().st_mtime)


def extract_text_from_pdf(file_path: Path) -> str:
    """извлекает текст из pdf с каждой стр"""
    try:
        reader = PdfReader(file_path)
    except Exception as exc:
        raise ValueError(f"Не удалось открыть PDF-файл {file_path}: {exc}") from exc

    pages_text = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages_text.append(page_text.strip())
        else:
            pages_text.append(f"[Страница {page_number} не содержит извлекаемого текста]")

    return "\n\n".join(pages_text).strip()


def find_cyrillic_font() -> Optional[str]:
    """ищет на системе шрифт с поддержкой кириллицы для fpdf"""
    candidates = [
        Path("DejaVuSans.ttf"),
        Path("/Library/Fonts/Arial Unicode.ttf"),
        Path("/Library/Fonts/AppleGothic.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/freefont/FreeSerif.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def save_to_pdf(report_text: str, filename: str) -> Path:
    """сохраняет отчет в pdf с оформлением"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    font_path = find_cyrillic_font()
    if font_path:
        pdf.add_font("DejaVu", "", font_path)
        pdf.set_font("DejaVu", size=12)
    else:
        pdf.set_font("Arial", size=12)

    paragraphs = report_text.split("\n\n")
    for paragraph in paragraphs:
        if paragraph.strip().startswith("===") and paragraph.strip().endswith("==="):
            # заголовок секции
            header_text = paragraph.strip().strip("=").strip()
            pdf.set_font("Arial", "B", 14)  
            pdf.cell(0, 10, header_text.upper(), ln=True)  
            pdf.set_draw_color(0, 0, 0) 
            pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 190, pdf.get_y())
            pdf.ln(5)
            pdf.set_font("Arial", size=12) 
        else:
            pdf.multi_cell(0, 10, text=paragraph)
            pdf.ln(2)

    # нумерация
    pdf.alias_nb_pages()
    pdf.set_y(-15)
    pdf.set_font("Arial", "I", 8)
    pdf.cell(0, 10, "Page " + str(pdf.page_no()) + "/{nb}", 0, 0, "C")

    output_path = OUTPUT_DIR / f"{filename}.pdf"
    pdf.output(str(output_path))
    return output_path


def load_api_key() -> str:
    """API ключ сначала из .env, затем из системного окружения"""
    # в .env GROQ_API_KEY
    for key in API_ENV_KEYS:
        value = os.getenv(key)
        if value:
            return value
    raise EnvironmentError(
        "API-ключ не найден. Поместите переменную GROQ_API_KEY в файл .env или установите ее в окружении. "
        + "Допустимые имена: " + ", ".join(API_ENV_KEYS)
    )


def build_report(document_text: str, previous_analysis_text: Optional[str], api_key: str) -> str:
    """запускаем многоагентный анализ через CrewAI"""
    llm = LLM(
        provider="groq",
        model=MODEL_NAME,
        api_key=api_key,
        temperature=0.0,
        max_tokens=4000,
    )

    business_analyst = Agent(
        role="Бизнес-аналитик",
        goal="Анализ ТЗ: бизнес-риски, информбезопасность, законодательство Казахстана.",
        backstory="Аналитик. Оценка рисков. Без выдумок. Только ТЗ.",
        llm=llm,
    )

    tech_lead = Agent(
        role="Техлид",
        goal="Техническая архитектура: нагрузка, ClickHouse, асинхрон, безопасность.",
        backstory="Техлид. Высоконагруженные системы. Без выдумок. Только ТЗ.",
        llm=llm,
    )

    task_business = Task(
        name="business-analysis",
        description=(
            "Анализ ТЗ:\n"
            "- Бизнес-риски\n"
            "- Аудитория, ценность\n"
            "- MVP, приоритеты\n"
            "- Сложность\n"
            "- Инфобезопасность\n"
            "- Законодательство РК\n"
            "- Вопросы для уточнения (3-5)\n\n"
            "Текст ТЗ: {document_text}\n"
            "Предварительный анализ: {previous_analysis_text}"
        ),
        expected_output=(
            "Структурированный отчет на русском. Риски, MVP, сложность, безопасность, законы, вопросы."
        ),
        agent=business_analyst,
    )

    task_technical = Task(
        name="technical-review",
        description=(
            "Дополнение к бизнес-анализу:\n"
            "- Архитектура, стек\n"
            "- ClickHouse для данных\n"
            "- Асинхронные сервисы\n"
            "- Риски, узкие места\n"
            "- Тестирование, CI/CD, Deploy"
        ),
        expected_output=(
            "Техническое дополнение. Стек, архитектура, ClickHouse, асинхрон, риски, CI/CD."
        ),
        agent=tech_lead,
        context=[task_business],
    )

    crew = Crew(
        name="tz-analysis-crew",
        process=Process.sequential,
        agents=[business_analyst, tech_lead],
        tasks=[task_business, task_technical],
    )

    time.sleep(5)
    crew_output = crew.kickoff(
        inputs={
            "document_text": document_text,
            "previous_analysis_text": previous_analysis_text or "Предыдущий анализ отсутствует",
        }
    )

    output_sections = []

    for task_output in crew_output.tasks_output:
        header = task_output.name or task_output.description[:60]
        body = (task_output.raw or "").strip()
        output_sections.append(f"=== {header} ({task_output.agent}) ===\n{body}\n")

    result = "\n".join(output_sections).strip()
    return result or crew_output.raw


def main() -> None:
    ensure_directories()

    input_pdf = locate_latest_pdf(INPUT_DIR)
    if not input_pdf:
        raise FileNotFoundError(
            "в папке inputs нет PDF-файлов. Поместите последний ТЗ в inputs и повторите запуск"
        )

    previous_pdf = locate_latest_pdf(OUTPUT_DIR)
    previous_analysis_text = None
    if previous_pdf:
        previous_analysis_text = extract_text_from_pdf(previous_pdf)

    api_key = load_api_key()
    document_text = extract_text_from_pdf(input_pdf)

    if not document_text:
        raise ValueError(f"Файл {input_pdf} не содержит извлекаемого текста.")

    report_text = build_report(document_text, previous_analysis_text, api_key)
    if not report_text.strip():
        raise RuntimeError("CrewAI вернул пустой отчет. Проверьте конфигурацию модели и ключ API.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = save_to_pdf(report_text, f"tz_analysis_{timestamp}")

    print(f"Входной файл: {input_pdf}")
    if previous_pdf:
        print(f"Сравнение выполнено с предыдущим отчетом: {previous_pdf}")
    print(f"Готовый отчет сохранен в: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Ошибка: {error}")
        raise
