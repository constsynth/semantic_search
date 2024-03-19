import torch
import typing as tp
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from tqdm import tqdm
import gc


class Searcher:
    def __init__(self,
                 model_name_or_path: str = 'intfloat/multilingual-e5-base',
                 number_of_data: int = 500,
                 create_database: bool = False):

        self.model = SentenceTransformer(model_name_or_path=model_name_or_path,
                                         device='cuda:0' if torch.cuda.is_available() else 'cpu')
        if create_database:
            self.documents = self.create_documents(number_of_data=number_of_data)
            self.document_embeddings = self.vectorize(self.documents)
            self.cleanup_memory()
        else:
            self.documents = []
            self.document_embeddings = None
            self.cleanup_memory()

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
            docs.append(dataset['train'][i]['text'])
        return docs

    def update_docs(self, new_doc: str = None) -> tp.NoReturn:
        self.documents.append(new_doc)
        embeddings = self.vectorize(new_doc)
        if self.document_embeddings is not None:
            self.document_embeddings = torch.cat((self.document_embeddings, embeddings), dim=0)
        else:
            self.document_embeddings = embeddings
        self.cleanup_memory()

    def vectorize(self,
                  documents: list[str] | str = None) -> torch.Tensor:
        print('\nDatabase embeddings creating...')
        documents_embeddings = []
        if isinstance(documents, list):
            for document in documents:
                document_embedding = self.model.encode(sentences=document,
                                                       device='cuda:0',
                                                       convert_to_tensor=True,
                                                       normalize_embeddings=True)
                document_embedding = torch.unsqueeze(document_embedding, 0)
                documents_embeddings.append(document_embedding)
            documents_embeddings = torch.cat(documents_embeddings)

        elif isinstance(documents, str):
            documents_embeddings = self.model.encode(sentences=documents,
                                                     device='cuda:0',
                                                     convert_to_tensor=True,
                                                     normalize_embeddings=True)
            documents_embeddings = torch.unsqueeze(documents_embeddings, 0)

        else:
            raise TypeError('documents must be a list of str or str')

        self.cleanup_memory()
        return documents_embeddings

    def find_closest_results(self, query: str = None,
                             first_n_closest: int = 5) -> tp.List[str]:
        user_query_embedding = self.model.encode(query, device='cuda:0', convert_to_tensor=True)
        user_query_embedding = torch.unsqueeze(user_query_embedding, 0)
        similarities = cosine_similarity(user_query_embedding.cpu().numpy(),
                                         self.document_embeddings.cpu().numpy()).tolist()
        similarities = {idx: score for idx, score in enumerate(similarities[0])}
        similarities_sorted_dict = dict(sorted(similarities.items(), key=lambda item: item[1], reverse=True))
        closest_results_indexes = list(similarities_sorted_dict.keys())[:first_n_closest]
        closest_results = [self.documents[closest_results_index] for closest_results_index in closest_results_indexes]
        self.cleanup_memory()
        return closest_results
