import hnswlib
import numpy as np
from sentence_transformers import SentenceTransformer
import time
import uuid
import json
import os

from models import (
    ChatGPT
)


class NuxAI:
    def __init__(self, model, api_key):
        self.llm = model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.api_key = api_key
        # edit this when we have other models
        self.supported_models = {
            'chatgpt': ChatGPT(api_key)
        }
    
    def _append_to_json_file(file_path, obj):
        data = []
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    pass
        data.append(obj)
        with open(file_path, 'w') as f:
            json.dump(data, f)


    def _embed(self, text):
        return self.embedding_model.encode(text)
    
    def _init_index(self, max_elements: int = 10000):
        strings = ['This is an example sentence']
        embeddings = self.embedding_model.encode(strings)
        dimension = embeddings.shape[1]

        # Initialize an index - the maximum number of elements should be known beforehand
        self.index = hnswlib.Index(space='cosine', dim=dimension)
        self.index.init_index(max_elements=max_elements, ef_construction=200, M=16)


    def optimize(self, max_combinations, user_prompts, desired_output):
        # set everything up
        model_handler = self.supported_models[self.llm]
        self._init_index(len(user_prompts))

        # generate N number of hyperparameter combinations
        param_combos = model_handler.get_hyperparameter_combinations(max_combinations)

        # each param and user prompt pair will generate a new response and embedding
        for params in param_combos:
            for prompt in user_prompts:
                response = model_handler.generate(prompt, params)
                obj = {
                    'id': str(uuid.uuid4()),
                    'user_prompt': prompt,
                    'hyperparams': params,
                    'model_response': response,
                    'embedding': self._embed(response)
                }
                # store the embedding in the index
                self.index.add_items(obj['embedding'])
                # store in a json
                self._append_to_json_file('data.json', obj)
        
        # query the index for the closest response
        query_embedding = self._embed(desired_output)
        ids, distances = self.index.knn_query(query_embedding, k=1)
        return ids, distances

