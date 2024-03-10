from transformers import BertModel, BertTokenizer
import torch
import typing as tp
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
from tqdm import tqdm
import gc


class Searcher:
    def __init__(self,
                 model_name_or_path: str = 'DeepPavlov/rubert-base-cased',
                 number_of_data: int = 500):
        self.device = torch.device('cuda:0')
        self.documents = self.create_documents(number_of_data=number_of_data)
        self.model, self.tokenizer = self.init_model(model_name_or_path)
        self.model.to(self.device)
        self.document_embeddings = self.vectorize_documents(self.documents)

    @staticmethod
    def cleanup_memory():
        torch.cuda.empty_cache()
        gc.collect()

    @staticmethod
    def create_documents(dataset_name: str = 'IlyaGusev/gazeta',
                         number_of_data: int = 20):
        dataset = load_dataset(dataset_name)
        docs = []
        print(f'Dataset is {dataset}')
        print('Documents list creating...')
        for i in tqdm(range(number_of_data)):
            docs.append(dataset['test'][i]['text'])
        return docs

    @staticmethod
    def init_model(model_name_or_path: str = None):
        model = BertModel.from_pretrained(model_name_or_path)
        tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
        return model, tokenizer

    def vectorize_documents(self,
                            documents: tp.List[str]) -> torch.Tensor:
        document_embeddings = []
        print('\nDatabase embeddings creating...')
        for document in tqdm(documents):
            inputs = self.tokenizer(document,
                                    return_tensors="pt",
                                    padding=True,
                                    truncation=True,
                                    max_length=128)
            outputs = self.model(**inputs.to(self.device))
            document_embedding = outputs.last_hidden_state.mean(dim=1)
            document_embeddings.append(document_embedding.detach().cpu())
        document_embeddings = torch.cat(document_embeddings)
        self.cleanup_memory()
        return document_embeddings

    def find_closest_results(self, query: str = None,
                             first_n_closest: int = 5) -> tp.List[str]:
        user_query_inputs = self.tokenizer(query,
                                           return_tensors="pt",
                                           padding=True,
                                           max_length=128)
        user_query_outputs = self.model(**user_query_inputs.to(self.device))
        user_query_embedding = user_query_outputs.last_hidden_state.mean(dim=1)

        similarities = cosine_similarity(user_query_embedding.detach().cpu(),
                            self.document_embeddings).tolist()
        similarities = {idx: score for idx, score in enumerate(similarities[0])}
        similarities_sorted_dict = dict(sorted(similarities.items(), key=lambda item: item[1], reverse=True))
        closest_results_indexes = list(similarities_sorted_dict.keys())[:first_n_closest]
        closest_results = [self.documents[closest_results_index] for closest_results_index in closest_results_indexes]
        self.cleanup_memory()
        return closest_results
