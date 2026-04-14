from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
from LLM import LLMClient 

class RAGEvaluationMetrics(BaseModel):    
    model_config = ConfigDict(extra='forbid')

    reasoning: str = Field(
        ...,
        description="Обоснование оценки одной строкой (без переносов). "
                    "Объясни, почему поставлены именно такие баллы."
    )
    
    faithfulness: int = Field(
        ...,
        description="Насколько ответ соответствует предоставленному контексту (1-5)",
        ge=1,
        le=5
    )
    
    relevance: int = Field(
        ...,
        description="Насколько ответ релевантен и полезен для исходного запроса (1-5)",
        ge=1,
        le=5
    )
    
    completeness: int = Field(
        ...,
        description="Насколько полно и всесторонне раскрыты факты и детали (1-5)",
        ge=1,
        le=5
    )
    

class LLMJudge:
    def __init__(self, client: LLMClient):
        self.client = client

    def evaluate(
        self, 
        query: str, 
        context: str, 
        response: str, 
        ground_truth: Optional[str] = None
    ) -> RAGEvaluationMetrics:
        
        system_prompt = (
            "Ты — строгий эксперт-аудитор RAG-систем. "
            "Оценивай ответ по четырём метрикам от 1 до 5.\n"
            "Будь объективен, точен и лаконичен.\n"
            "reasoning пиши одной строкой без переносов."
        )

        user_prompt = (
            f"ВОПРОС: {query}\n\n"
            f"КОНТЕКСТ:\n{context}\n\n"
            f"ЭТАЛОН (если есть): {ground_truth or 'Не предоставлен'}\n\n"
            f"ОТВЕТ СИСТЕМЫ:\n{response}\n\n"
            "Оцени строго по схеме RAGEvaluationMetrics."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        result = self.client.chat(
            messages=messages,
            response_format=RAGEvaluationMetrics 
        )

        if isinstance(result, dict):
            return RAGEvaluationMetrics.model_validate(result)
        elif isinstance(result, str):
            return RAGEvaluationMetrics.model_validate_json(result)
        else:
            raise ValueError(f"Неожиданный тип ответа от LLMClient: {type(result)}")