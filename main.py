import hnswlib
import numpy as np
from sentence_transformers import SentenceTransformer
import time

from models import (
    ChatGPT
)


class NuxAI:
    def __init__(self, model, api_key):
        self.llm = model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.api_key = api_key
        # edit this with other models
        self.supported_models = {
            'chatgpt': ChatGPT(api_key)
        }

    def _embed(self, text):
        return self.embedding_model.encode(text)
    
    def _init_index(self, max_elements: int = 10000):
        strings = ['This is an example sentence']
        embeddings = self.embedding_model.encode(strings)
        dimension = embeddings.shape[1]

        # Initialize an index - the maximum number of elements should be known beforehand
        self.index = hnswlib.Index(space='cosine', dim=dimension)
        self.index.init_index(max_elements=max_elements, ef_construction=200, M=16)


    def optimize(self, interval, user_prompts, desired_output):
        # set everything up
        model_handler = self.supported_models[self.llm]
        self._init_index(len(user_prompts))


        response = model_handler.generate(user_prompts[0], {'temperature': 0.1})

        # Generate hyperparameter combinations (for simplicity, using only 'temperature' in this example)
        temperature_range = [round(x * interval, 1) for x in range(1, int(1/interval) + 1)]
        
        results = []
        for prompt in user_prompts:
            for temp in temperature_range:
                hyperparams = {'temperature': temp}
                response = self.generate_mock_response(prompt, hyperparams)
                embedding = self.sentence_model.encode(response)
                idx = len(results)  # Unique index for each response
                self.hnsw_index.add_items(embedding, idx)
                results.append({'prompt': prompt, 'hyperparams': hyperparams, 'response': response})

        # Embedding for desired output
        desired_embedding = self.sentence_model.encode(desired_output)
        labels, distances = self.hnsw_index.knn_query(desired_embedding, k=len(results))

        # Organizing results
        optimized_results = []
        for label, distance in zip(labels[0], distances[0]):
            optimized_results.append({
                'prompt': results[label]['prompt'],
                'hyperparams': results[label]['hyperparams'],
                'response': results[label]['response'],
                'similarity': 1 - distance  # Cosine similarity
            })

        return optimized_results

# Example usage
nux = NuxAI(model="chatgpt", api_key="***")
user_prompts = [
    "generate a summary from this article: {{article}}", 
    "take a deep breath and generate a
