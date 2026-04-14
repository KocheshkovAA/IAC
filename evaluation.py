import json
import numpy as np
from LLMJudge import LLMJudge
from RAG import RAGSystem

def run_evaluation(rag: RAGSystem, dataset_path: str):
    judge = LLMJudge(rag.generator.llm) 
    report = []

    dataset = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line: 
                dataset.append(json.loads(line))

    for item in dataset:
        query = item.get('user_query') or item.get('question')
        ground_truth = item.get('ground_truth', "")
        
        print(f"\n🧐 Запрос: {query}")
        
        relevant_chunks = rag.db.search(query, top_k=3)
        context_text = "\n\n".join([c['text'] for c in relevant_chunks])

        rag_output = rag.query(query)
        if not rag_output:
            print("❌ Система не смогла сгенерировать ответ.")
            continue

        print(f"\nОтвет: {rag_output.final_regulation}")

        try:
            eval_res = judge.evaluate(
                query=query, 
                context=context_text, 
                response=rag_output.final_regulation,
                ground_truth=ground_truth
            )
            
            print(f"📊 F: {eval_res.faithfulness} | R: {eval_res.relevance} | C: {eval_res.completeness}")
            print(f"📝 Обоснование: {eval_res.reasoning}")
            
            report.append(eval_res.model_dump())
        except Exception as e:
            print(f"❌ Ошибка при оценке судьёй: {e}")

    if report:
        f_scores = [r['faithfulness'] for r in report]
        r_scores = [r['relevance'] for r in report]
        c_scores = [r['completeness'] for r in report]

        print("\n" + "="*40)
        print("📈 ИТОГОВЫЕ МЕТРИКИ (SRG EVAL)")
        print(f"Faithfulness (Верность):  {np.mean(f_scores):.2f} / 5.0")
        print(f"Relevance (Релевантность): {np.mean(r_scores):.2f} / 5.0")
        print(f"Completeness (Полнота):    {np.mean(c_scores):.2f} / 5.0")
        
        total_avg = np.mean(f_scores + r_scores + c_scores)
        print(f"\n🏆 ОБЩИЙ SCORE СИСТЕМЫ: {total_avg:.2f}")
        print("="*40)
    else:
        print("📭 Отчет пуст. Проверь корректность данных.")
