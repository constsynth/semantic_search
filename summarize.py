from transformers import AutoModelForSeq2SeqLM, MBartTokenizer
import torch
import gc


class Summarizer:

    def __init__(self,
                 model_name_or_path: str = ''
                 ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model, self.tokenizer = self.model_init(model_name_or_path)
        self.model.to(self.device)
        self.model.to_bettertransformer()

    @staticmethod
    def model_init(model_name: str = None):

        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = MBartTokenizer.from_pretrained(model_name)
        return model, tokenizer

    @staticmethod
    def cleanup_memory():
        torch.cuda.empty_cache()
        gc.collect()

    def summarize(self,
                  text: str = None,
                  max_input: int = 512,
                  max_length: int = 200,
                  min_length: int = 64,
                  no_repeat_ngram_size: int = 4,
                  temperature: float = 1.0
                  ) -> str:

        encode = self.tokenizer(
            [text],
            padding='max_length',
            truncation=True,
            max_length=max_input,
            return_tensors='pt'
        )

        input_ids = encode['input_ids'].to(self.device)
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=False):
            predicts = self.model.generate(input_ids,
                                           max_length=max_length,
                                           min_length=min_length,
                                           no_repeat_ngram_size=no_repeat_ngram_size,
                                           temperature=temperature
                                           )
        for output in predicts:
            summary = self.tokenizer.decode(output, skip_special_tokens=True)
        self.cleanup_memory()
        return summary
