import json
import re
from typing import List, Dict
from LLM import LLMClient
from pydantic import BaseModel, Field

class SocialInstructionSGR(BaseModel):
    extracted_entities: List[str] = Field(
        description="Список всех найденных в контексте дат, сроков, сумм, процентов и названий организаций."
    )
    analysis: str = Field(
        description="Глубокий анализ прав и условий на основе извлеченных фактов."
    )
    steps: List[str] = Field(
        description="Пошаговый алгоритм действий гражданина (с указанием конкретных ведомств и сроков)."
    )
    final_regulation: str = Field(
        description="Итоговый официальный текст инструкции."
    )

class GenerationModule:
    def __init__(self, client: LLMClient):
        self.llm = client

    def generate_instruction(self, query: str, context_chunks: List[Dict]) -> SocialInstructionSGR:
        context_text = "\n\n".join([chunk['text'] for chunk in context_chunks])

        system_prompt = (
            "Ты — эксперт-аналитик по социальным вопросам. Твоя задача — составить инструкцию на основе предоставленного текста.\n\n"
            "ПРАВИЛА РАБОТЫ:\n"
            "1. Используй ТОЛЬКО информацию из КАНТЕКСТА. Если данных нет, пиши 'не указано'.\n"
            "2. extracted_entities: Строгий список ключевых фактов (названия ГКУ/МФЦ, суммы в рублях, сроки в рабочих днях).\n"
            "3. analysis: Обоснуй право на услугу (нормативная база из текста, критерии заявителя).\n"
            "4. steps: Четкий алгоритм: Куда -> С какими документами -> Время ожидания -> Ожидаемый результат.\n"
            "5. final_regulation: Официальный, структурированный текст инструкции. Используй факты из 'extracted_entities'.\n\n"
            "ОТВЕТЬ СТРОГО В JSON."
            "ЭТАЛОН СТРУКТУРЫ:\n"
            "{\n"
            "  \"extracted_entities\": [\"СПб ГКУ 'ЦОСО'\", \"30 дней\"],\n"
            "  \"analysis\": \"Услуга доступна ветеранам на основании...\",\n"
            "  \"steps\": [\"Шаг 1: Собрать документы\", \"Шаг 2: Подать заявление\"],\n"
            "  \"final_regulation\": \"Полный текст инструкции...\"\n"
            "}"
        )

        user_prompt = (
            f"КОНТЕКСТ ДЛЯ АНАЛИЗА:\n{context_text}\n\n"
            f"ЗАПРОС ПОЛЬЗОВАТЕЛЯ: {query}\n\n"
            "Сгенерируй JSON объект согласно схеме SocialInstructionSGR:"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        raw_output = self.llm.chat(
            messages=messages,
            response_format={"type": "json_object"}
        )

        clean_json = raw_output.strip().replace("```json", "").replace("```", "").strip()
        clean_json = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', clean_json)
        
        try:
            return SocialInstructionSGR.model_validate_json(clean_json)
        except Exception:
            data = json.loads(clean_json, strict=False)
            return SocialInstructionSGR(
                extracted_entities=data.get("extracted_entities", []) or data.get("facts", []),
                analysis=data.get("analysis", "Ошибка разбора анализа"),
                steps=data.get("steps", []) or ["Данные не извлечены"],
                final_regulation=data.get("final_regulation") or data.get("reglament") or "Ошибка формирования итогового текста"
            )