from pydantic import BaseModel, Field, AliasChoices
import re

class RAGEvaluationMetrics(BaseModel):
    reasoning: str = Field(
        default="Обоснование не предоставлено моделью",
        validation_alias=AliasChoices(
            'reasoning', 'Reasoning', 'analysis', 'Analysis', 
            'explanation', 'Explanation', 'обоснование', 'Обоснование'
        )
    )
    faithfulness: int = Field(
        validation_alias=AliasChoices('faithfulness', 'Faithfulness', 'верность', 'Верность'),
        ge=1, le=5
    )
    relevance: int = Field(
        validation_alias=AliasChoices('relevance', 'Relevance', 'релевантность', 'Релевантность'),
        ge=1, le=5
    )
    completeness: int = Field(
        validation_alias=AliasChoices('completeness', 'Completeness', 'полнота', 'Полнота'),
        ge=1, le=5
    )
    
class LLMJudge:
    def __init__(self, client):
        self.client = client

    def evaluate(self, query: str, context: str, response: str, ground_truth: str = None) -> RAGEvaluationMetrics:
        system_prompt = (
            "Ты — эксперт-аудитор. Оцени ответ RAG-системы по шкале от 1 до 5.\n"
            "ПРАВИЛА:\n"
            "1. reasoning: Пиши обоснование ОДНОЙ строкой, без переносов (Enter).\n"
            "2. faithfulness: Насколько ответ соответствует контексту.\n"
            "3. relevance: Насколько ответ полезен на запрос.\n"
            "4. completeness: Насколько полно даны факты.\n"
            "Верни СТРОГО JSON без вводных фраз."
        )

        user_prompt = f"ВОПРОС: {query}\nКОНТЕКСТ: {context}\nЭТАЛОН: {ground_truth or 'Не указан'}\nОТВЕТ: {response}"

        raw_output = self.client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )

        clean_json = raw_output.strip().replace("```json", "").replace("```", "").strip()
        
        clean_json = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', clean_json)
        
        return RAGEvaluationMetrics.model_validate_json(clean_json)