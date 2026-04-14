import requests
from bs4 import BeautifulSoup
import re

def clean_text(text):
    if not text: return ""
    text = text.replace('\r', ' ').replace('\n', ' ').replace('\t', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = text.replace(" ( ", " (").replace(" ) ", ") ").replace(" .", ".").replace(" ,", ",")
    return text.strip()

def parse_gu_knowledge_base(url_or_html, is_url=True):
    if is_url:
        response = requests.get(url_or_html)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')
    else:
        soup = BeautifulSoup(url_or_html, 'html.parser')

    for hidden in soup.find_all(class_='visually-hidden'):
        hidden.decompose()

    all_chunks = []
    articles = soup.find_all('article', class_='droppanel')
    
    for article in articles:
        header = article.find('span', class_='droppanel__head-title')
        section_title = header.get_text(strip=True) if header else "Общий раздел"
        body = article.find('div', class_='droppanel__frame')
        if not body: continue

        for a in body.find_all('a', href=True):
            a_text = a.get_text(strip=True)
            href = a['href'].strip()
            if href.startswith('/'): href = f"https://gu.spb.ru{href}"
            a.replace_with(f" {a_text} ({href}) ")

        current_question = None
        current_answer_text = ""

        for element in body.find_all(recursive=False):
            raw_el_text = element.get_text(" ", strip=True)
            if not raw_el_text: continue

            is_question = (element.name == 'p' and 
                          (element.find('strong') or raw_el_text.endswith('?')))

            if is_question:
                if current_question:
                    all_chunks.append({
                        "text": f"РАЗДЕЛ: {section_title}\nВОПРОС: {current_question}\nОТВЕТ: {clean_text(current_answer_text)}",
                        "metadata": {"section": section_title, "question": current_question}
                    })
                
                current_question = clean_text(raw_el_text)
                current_answer_text = ""
            else:
                if element.name in ['ol', 'ul']:
                    list_items = []
                    for i, li in enumerate(element.find_all('li', recursive=False), 1):
                        prefix = f"{i}." if element.name == 'ol' else "•"
                        list_items.append(f"{prefix} {li.get_text(' ', strip=True)}")
                    current_answer_text += "\n" + "\n".join(list_items) + "\n"
                else:
                    current_answer_text += " " + raw_el_text

        if current_question:
            all_chunks.append({
                "text": f"РАЗДЕЛ: {section_title}\nВОПРОС: {current_question}\nОТВЕТ: {clean_text(current_answer_text)}",
                "metadata": {"section": section_title, "question": current_question}
            })

    return all_chunks