import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RAG-Engine")

class RAGSystem:    
    def __init__(
        self, 
        vector_db, 
        generator_module, 
        index_path: str = 'knowledge_base.index',
        meta_path: str = 'metadata.json'
    ):
        self.db = vector_db
        self.generator = generator_module
        self.index_path = index_path
        self.meta_path = meta_path

    def prepare_knowledge_base(self, parser_func, target_url: str = "https://gu.spb.ru/knowledge-base/zhitelyu-blokadnogo-leningrada/"):
        if os.path.exists(self.index_path):
            self.db.load(self.index_path, self.meta_path)
        else:
            chunks = parser_func(target_url, is_url=True)
            self.db.build_index(chunks)
            self.db.save(self.index_path, self.meta_path)

    def query(self, user_query: str, top_k: int = 3):

        relevant_chunks = self.db.search(user_query, top_k=top_k)

        try:
            return self.generator.generate_instruction(user_query, relevant_chunks)
        except Exception as e:
            logger.error(f"❌ Ошибка генерации: {e}")
            return None