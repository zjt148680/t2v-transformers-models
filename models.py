import torch
from transformers import AutoTokenizer, AutoModel, DPRQuestionEncoder, DPRContextEncoder, T5ForConditionalGeneration, \
    T5Tokenizer

from RequestParams import VectorInputConfig

DEFAULT_POOL_METHOD = "masked_mean"


class ModelFactory:

    @staticmethod
    def model(
            model_type,
            architecture,
            cuda_support: bool,
            cuda_core: str,
            trust_remote_code: bool,
    ):
        if model_type == "t5":
            return T5Model(cuda_support, cuda_core, trust_remote_code)
        elif model_type == "dpr":
            return DPRModel(architecture, cuda_support, cuda_core, trust_remote_code)
        elif model_type == "qwen3":
            return Qwen3Model(
                cuda_support,
                cuda_core,
                trust_remote_code,
            )
        else:
            return HFModel(cuda_support, cuda_core, trust_remote_code)


class HFModel:
    def __init__(self, cuda_support: bool, cuda_core: str, trust_remote_code: bool):
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.cuda = cuda_support
        self.cuda_core = cuda_core
        self.trust_remote_code = trust_remote_code

    def create_tokenizer(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=self.trust_remote_code
        )
        return self.tokenizer

    def create_model(self, model_path):
        self.model = AutoModel.from_pretrained(
            model_path, trust_remote_code=self.trust_remote_code
        )
        return self.model

    def get_embeddings(self, batch_results):
        return batch_results[0]

    def get_batch_results(self, tokens, text):
        return self.model(**tokens)

    def pool_embedding(self, batch_results, tokens, config: VectorInputConfig):
        pooling_method = self.pool_method_from_config(config)
        if pooling_method == "cls":
            return self.get_embeddings(batch_results)[:, 0, :]
        elif pooling_method == "masked_mean":
            return self.pool_sum(
                self.get_embeddings(batch_results), tokens["attention_mask"]
            )
        else:
            raise Exception(f"invalid pooling method '{pooling_method}'")

    def pool_method_from_config(self, config: VectorInputConfig):
        if config is None:
            return DEFAULT_POOL_METHOD

        if config.pooling_strategy is None or config.pooling_strategy == "":
            return DEFAULT_POOL_METHOD

        return config.pooling_strategy

    def get_sum_embeddings_mask(self, embeddings, input_mask_expanded):
        if self.cuda:
            sum_embeddings = torch.sum(embeddings * input_mask_expanded, 1).to(
                self.cuda_core
            )
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9).to(
                self.cuda_core
            )
            return sum_embeddings, sum_mask
        else:
            sum_embeddings = torch.sum(embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings, sum_mask

    def pool_sum(self, embeddings, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        )
        sum_embeddings, sum_mask = self.get_sum_embeddings_mask(
            embeddings, input_mask_expanded
        )
        sentences = sum_embeddings / sum_mask
        return sentences


class DPRModel(HFModel):

    def __init__(
            self,
            architecture: str,
            cuda_support: bool,
            cuda_core: str,
            trust_remote_code: bool,
    ):
        super().__init__(cuda_support, cuda_core, trust_remote_code)
        self.model = None
        self.architecture = architecture
        self.trust_remote_code = trust_remote_code

    def create_model(self, model_path):
        if self.architecture == "DPRQuestionEncoder":
            self.model = DPRQuestionEncoder.from_pretrained(
                model_path, trust_remote_code=self.trust_remote_code
            )
        else:
            self.model = DPRContextEncoder.from_pretrained(
                model_path, trust_remote_code=self.trust_remote_code
            )
        return self.model

    def get_batch_results(self, tokens, text):
        return self.model(tokens["input_ids"], tokens["attention_mask"])

    def pool_embedding(self, batch_results, tokens, config: VectorInputConfig):
        # no pooling needed for DPR
        return batch_results["pooler_output"][0]


class T5Model(HFModel):

    def __init__(self, cuda_support: bool, cuda_core: str, trust_remote_code: bool):
        super().__init__(cuda_support, cuda_core)
        self.model = None
        self.tokenizer = None
        self.cuda = cuda_support
        self.cuda_core = cuda_core
        self.trust_remote_code = trust_remote_code

    def create_model(self, model_path):
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_path, trust_remote_code=self.trust_remote_code
        )
        return self.model

    def create_tokenizer(self, model_path):
        self.tokenizer = T5Tokenizer.from_pretrained(
            model_path, trust_remote_code=self.trust_remote_code
        )
        return self.tokenizer

    def get_embeddings(self, batch_results):
        return batch_results["encoder_last_hidden_state"]

    def get_batch_results(self, tokens, text):
        input_ids, attention_mask = tokens["input_ids"], tokens["attention_mask"]

        target_encoding = self.tokenizer(
            text, padding="longest", max_length=500, truncation=True
        )
        labels = target_encoding.input_ids
        if self.cuda:
            labels = torch.tensor(labels).to(self.cuda_core)
        else:
            labels = torch.tensor(labels)

        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )


class Qwen3Model(HFModel):
    def __init__(self, cuda_support: bool, cuda_core: str, trust_remote_code: bool):
        super().__init__(cuda_support, cuda_core, trust_remote_code)

    def create_tokenizer(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=self.trust_remote_code, padding_side='left'
        )
        return self.tokenizer

    def get_attention_mask(self, tokens):
        return tokens["attention_mask"]

    def pool_embedding(self, batch_results, tokens, config: VectorInputConfig):
        last_hidden_state = self.get_embeddings(batch_results)

        attention_mask = torch.tensor(tokens["attention_mask"]).to(self.cuda_core)
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_state[:, -1]
        else:
            sequence_lengths = last_hidden_state.sum(dim=1) - 1
            batch_size = last_hidden_state.shape[0]
            return last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
