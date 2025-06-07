import asyncio
from concurrent.futures import ThreadPoolExecutor
from logging import getLogger, Logger
from pathlib import Path
from typing import Any, Literal, List

import nltk
import torch
import torch.nn.functional as F
from cachetools import cached
from optimum.onnxruntime import ORTModelForFeatureExtraction
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModel,
    AutoTokenizer,
)

from RequestParams import VectorInputConfig
from config import get_cache_settings
from models import ModelFactory, HFModel

# limit transformer batch size to limit parallel inference, otherwise we run
# into memory problems
MAX_BATCH_SIZE = 25  # TODO: take from config
DEFAULT_LANGUAGE = "chinese"  # TODO: take from config


class Vectorizer:
    executor: ThreadPoolExecutor

    def __init__(
            self,
            model_path: str,
            cuda_support: bool,
            cuda_core: str,
            cuda_per_process_memory_fraction: float,
            model_type: str,
            architecture: str,
            direct_tokenize: bool,
            onnx_runtime: bool,
            use_sentence_transformers_vectorizer: bool,
            use_sentence_transformers_multi_process: bool,
            model_name: str,
            trust_remote_code: bool,
            workers: int | None,
    ):
        self.executor = ThreadPoolExecutor()
        if onnx_runtime:
            self.vectorizer = ONNXVectorizer(model_path, trust_remote_code)
        else:
            if model_type == "t5" or use_sentence_transformers_vectorizer:
                self.vectorizer = SentenceTransformerVectorizer(
                    model_path,
                    model_name,
                    cuda_core,
                    trust_remote_code,
                    use_sentence_transformers_multi_process,
                    workers,
                )
            elif model_type == "qwen3":
                self.vectorizer = Qwen3Vectorizer(
                    model_path,
                    cuda_support,
                    cuda_core,
                    cuda_per_process_memory_fraction,
                    model_type,
                    architecture,
                    direct_tokenize,
                    trust_remote_code,
                )
            else:
                self.vectorizer = HuggingFaceVectorizer(
                    model_path,
                    cuda_support,
                    cuda_core,
                    cuda_per_process_memory_fraction,
                    model_type,
                    architecture,
                    direct_tokenize,
                    trust_remote_code,
                )

    async def vectorize(self, text: str, config: VectorInputConfig, worker: int = 0):
        if isinstance(self.vectorizer, SentenceTransformerVectorizer):
            loop = asyncio.get_event_loop()
            f = loop.run_in_executor(
                self.executor, self.vectorizer.vectorize, text, config, worker
            )
            return await asyncio.wrap_future(f)

        return await asyncio.wrap_future(
            self.executor.submit(self.vectorizer.vectorize, text, config)
        )

    async def batch_vectorize(self, texts: List[str], config: VectorInputConfig, worker: int = 0):
        if isinstance(self.vectorizer, SentenceTransformerVectorizer):
            loop = asyncio.get_event_loop()
            f = loop.run_in_executor(
                self.executor, self.vectorizer.batch_vectorize, texts, config, worker
            )
            return await asyncio.wrap_future(f)

        return await asyncio.wrap_future(
            self.executor.submit(self.vectorizer.batch_vectorize, texts, config)
        )


class SentenceTransformerVectorizer:
    workers: List[SentenceTransformer]
    available_devices: List[str]
    cuda_core: str
    use_sentence_transformers_multi_process: bool
    pool: dict[Literal["input", "output", "processes"], Any]
    logger: Logger

    def __init__(
            self,
            model_path: str,
            model_name: str,
            cuda_core: str,
            trust_remote_code: bool,
            use_sentence_transformers_multi_process: bool,
            workers: int | None,
    ):
        self.logger = getLogger("uvicorn")
        self.cuda_core = cuda_core
        self.use_sentence_transformers_multi_process = (
            use_sentence_transformers_multi_process
        )
        self.available_devices = self.get_devices(
            workers, self.use_sentence_transformers_multi_process
        )
        self.logger.info(
            f"Sentence transformer vectorizer running with model_name={model_name}, cache_folder={model_path} trust_remote_code:{trust_remote_code}"
        )
        self.workers = []
        for device in self.available_devices:
            model = SentenceTransformer(
                model_name,
                cache_folder=model_path,
                device=device,
                trust_remote_code=trust_remote_code,
            )
            model.eval()  # make sure we're in inference mode, not training
            self.workers.append(model)

        if self.use_sentence_transformers_multi_process:
            self.pool = self.workers[0].start_multi_process_pool(
                target_devices=self.get_cuda_devices()
            )
            self.logger.info(
                "Sentence transformer vectorizer is set to use all available devices"
            )
            self.logger.info(
                f"Created pool of {len(self.pool['processes'])} available {'CUDA' if torch.cuda.is_available() else 'CPU'} devices"
            )

    def get_cuda_devices(self) -> List[str] | None:
        if self.cuda_core is not None and self.cuda_core != "":
            return self.cuda_core.split(",")

    def get_devices(
            self,
            workers: int | None,
            use_sentence_transformers_multi_process: bool,
    ) -> List[str | None]:
        if (
                not self.use_sentence_transformers_multi_process
                and self.cuda_core is not None
                and self.cuda_core != ""
        ):
            return self.cuda_core.split(",")
        if use_sentence_transformers_multi_process or workers is None or workers < 1:
            return [None]
        return [None] * workers

    @cached(cache=get_cache_settings())
    def vectorize(self, text: str, config: VectorInputConfig, worker: int = 0):
        if self.use_sentence_transformers_multi_process:
            embedding = self.workers[0].encode_multi_process(
                [text], pool=self.pool, normalize_embeddings=True
            )
            return embedding[0]

        embedding = self.workers[worker].encode(
            [text],
            device=self.available_devices[worker],
            convert_to_tensor=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding

    @cached(cache=get_cache_settings())
    def batch_vectorize(self, texts: List[str], config: VectorInputConfig, worker: int = 0):
        pass


class ONNXVectorizer:
    model: ORTModelForFeatureExtraction
    tokenizer: AutoTokenizer

    def __init__(self, model_path, trust_remote_code: bool) -> None:
        onnx_path = Path(model_path)
        self.model = ORTModelForFeatureExtraction.from_pretrained(
            onnx_path,
            file_name="model_quantized.onnx",
            trust_remote_code=trust_remote_code,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            onnx_path, trust_remote_code=trust_remote_code
        )

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def vectorize(self, text: str, config: VectorInputConfig):
        encoded_input = self.tokenizer(
            [text], padding=True, truncation=True, return_tensors="pt"
        )
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling
        sentence_embeddings = self.mean_pooling(
            model_output, encoded_input["attention_mask"]
        )

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings

    def batch_vectorize(self, texts: List[str], config: VectorInputConfig):
        pass


class HuggingFaceVectorizer:
    model: AutoModel
    tokenizer: AutoTokenizer
    cuda: bool
    cuda_core: str
    model_type: str
    direct_tokenize: bool
    trust_remote_code: bool

    def __init__(
            self,
            model_path: str,
            cuda_support: bool,
            cuda_core: str,
            cuda_per_process_memory_fraction: float,
            model_type: str,
            architecture: str,
            direct_tokenize: bool,
            trust_remote_code: bool,
    ):
        self.cuda = cuda_support
        self.cuda_core = cuda_core
        self.cuda_per_process_memory_fraction = cuda_per_process_memory_fraction
        self.model_type = model_type
        self.direct_tokenize = direct_tokenize
        self.trust_remote_code = trust_remote_code

        self.model_delegate: HFModel = ModelFactory.model(
            model_type, architecture, cuda_support, cuda_core, trust_remote_code
        )
        self.model = self.model_delegate.create_model(model_path)

        if self.cuda:
            self.model.to(self.cuda_core)
            if self.cuda_per_process_memory_fraction:
                torch.cuda.set_per_process_memory_fraction(
                    self.cuda_per_process_memory_fraction
                )
        self.model.eval()  # make sure we're in inference mode, not training

        self.tokenizer = self.model_delegate.create_tokenizer(model_path)

        nltk.data.path.append("./nltk_data")

    def tokenize(self, text: str | List[str]):
        return self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=500,
            add_special_tokens=True,
            return_tensors="pt",
        )

    def get_embeddings(self, batch_results):
        return self.model_delegate.get_embeddings(batch_results)

    def get_batch_results(self, tokens, text):
        return self.model_delegate.get_batch_results(tokens, text)

    def pool_embedding(self, batch_results, tokens, config):
        return self.model_delegate.pool_embedding(batch_results, tokens, config)

    def vectorize(self, text: str | List[str], config: VectorInputConfig):
        with torch.no_grad():
            # create embeddings without tokenizing text
            tokens = self.tokenize(text)
            if self.cuda:
                tokens.to(self.cuda_core)
            batch_results = self.get_batch_results(tokens, text)
            batch_sum_vectors = self.pool_embedding(batch_results, tokens, config)
            return batch_sum_vectors.detach()

    def batch_vectorize(self, texts: List[str], config: VectorInputConfig):
        return self.vectorize(texts, config)


class Qwen3Vectorizer(HuggingFaceVectorizer):
    model: AutoModel
    tokenizer: AutoTokenizer
    cuda: bool
    cuda_core: str
    model_type: str
    direct_tokenize: bool
    trust_remote_code: bool

    def __init__(
            self,
            model_path: str,
            cuda_support: bool,
            cuda_core: str,
            cuda_per_process_memory_fraction: float,
            model_type: str,
            architecture: str,
            direct_tokenize: bool,
            trust_remote_code: bool,
    ):
        super().__init__(
            model_path,
            cuda_support,
            cuda_core,
            cuda_per_process_memory_fraction,
            model_type,
            architecture,
            direct_tokenize,
            trust_remote_code,
        )

        self.eod_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        self.max_length = 8192

    def tokenize(self, text: str | List[str]):
        if isinstance(text, str):
            text = [text]
        batch_dict = self.tokenizer(text, padding=False, truncation=True, max_length=self.max_length - 2)
        for seq, att in zip(batch_dict["input_ids"], batch_dict["attention_mask"]):
            seq.append(self.eod_id)
            att.append(1)
        batch_dict = self.tokenizer.pad(batch_dict, padding=True, return_tensors="pt")
        return batch_dict
