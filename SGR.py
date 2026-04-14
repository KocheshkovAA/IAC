from typing import List, Dict
from pydantic import BaseModel, Field, ConfigDict
from LLM import LLMClient 


class SocialInstructionSGR(BaseModel):    
    model_config = ConfigDict(extra='forbid') 

    extracted_entities: List[str] = Field(
        ...,
        description="Список всех найденных в контексте дат, сроков, сумм, процентов и названий организаций/ведомств. "
                    "Только точные факты из текста. Если ничего нет — пустой список []."
    )
    
    analysis: str = Field(
        ...,
        description="Глубокий анализ права гражданина на услугу. "
                    "Обоснуй на основе нормативной базы и критериев из контекста."
    )
    
    steps: List[str] = Field(
        ...,
        description="Пошаговый алгоритм действий. Каждый шаг должен содержать: "
                    "Куда обратиться (название организации) → Какие документы → Срок → Ожидаемый результат."
    )
    
    final_regulation: str = Field(
        ...,
        description="Итоговый официальный, лаконичный и готовый к выдаче текст инструкции для гражданина. "
                    "Стиль — деловой, без лишних слов."
    )


class GenerationModule:
    def __init__(self, client: LLMClient):
        self.llm = client

    def generate_instruction(self, query: str, context_chunks: List[Dict]) -> SocialInstructionSGR:
        context_text = "\n\n".join(chunk['text'] for chunk in context_chunks)

        system_prompt = (
            "Ты — эксперт по социальным услугам и государственным инструкциям России.\n"
            "Отвечай ИСКЛЮЧИТЕЛЬНО на основе предоставленного контекста. Не придумывай факты.\n"
            "Строго следуй порядку мышления:\n"
            "1. Извлеки ключевые факты → 2. Проведи анализ права → 3. Составь пошаговый алгоритм → 4. Сформируй финальную инструкцию.\n"
            "Будь точным и лаконичным."
        )

        user_prompt = (
            f"КОНТЕКСТ:\n{context_text}\n\n"
            f"ЗАПРОС ГРАЖДАНИНА: {query}\n\n"
            "Сгенерируй ответ строго по схеме SocialInstructionSGR."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            result = self.llm.chat(
                messages=messages,
                response_format=SocialInstructionSGR,
            )

            if isinstance(result, dict):
                return SocialInstructionSGR.model_validate(result)
            elif isinstance(result, str):
                return SocialInstructionSGR.model_validate_json(result)
            else:
                raise ValueError(f"Неожиданный тип результата от LLMClient: {type(result)}")

        except Exception as e:
            raise RuntimeError(f"Ошибка генерации инструкции: {str(e)}") from e