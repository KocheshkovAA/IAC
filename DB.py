import faiss
import json
from sentence_transformers import SentenceTransformer

class VectorDB:
    def __init__(self, model_name='ai-forever/FRIDA'):

        self.model = SentenceTransformer(model_name, device='cpu')
        self.index = None
        self.metadata = []
        
    def _prepare_text(self, text, is_query=False):
        prefix = "search_query: " if is_query else "search_document: "
        return prefix + text

    def build_index(self, chunks):
        texts = [self._prepare_text(c['text']) for c in chunks]
        self.metadata = chunks
        
        embeddings = self.model.encode(texts, show_progress_bar=True)
        dimension = embeddings.shape[1]
        
        self.index = faiss.IndexFlatL2(dimension)

        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))

    def search(self, query, top_k=3):
        query_text = self._prepare_text(query, is_query=True)
        query_embedding = self.model.encode([query_text])
        faiss.normalize_L2(query_embedding)
        
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for idx in indices[0]:
            if idx != -1: 
                results.append(self.metadata[idx])
        
        return results

    def save(self, index_path='docs.index', meta_path='metadata.json'):
        faiss.write_index(self.index, index_path)
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False)

    def load(self, index_path='docs.index', meta_path='metadata.json'):
        self.index = faiss.read_index(index_path)
        with open(meta_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)