# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import torch
import json
import os
from cog import BasePredictor, Input, ConcatenateIterator
from transformers import AutoTokenizer
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams

# Smaller chunk size to avoid OOM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

MODEL_NAME = "TheBloke/OpenHermes-2-Mistral-7B-AWQ"
TOKENIZER_MODEL_NAME = "teknium/OpenHermes-2-Mistral-7B"
MODEL_CACHE = "model-cache"
TOKEN_CACHE = "token-cache"

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_MODEL_NAME,
            cache_dir=TOKEN_CACHE
        )
        args = AsyncEngineArgs(
            model=MODEL_NAME,
            tokenizer=TOKENIZER_MODEL_NAME,
            quantization="awq",
            dtype="float16",
            # gpu_memory_utilization=0.2
        )
        self.engine = AsyncLLMEngine.from_engine_args(args)


    def predict(
        self,
        prompt: str = Input(description="The JSON stringified of the messages (array of objects with role/content like OpenAI) to predict on"),
        max_new_tokens: int = Input(description="Max new tokens", ge=1, default=512),
        temperature: float = Input(
            description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic, 0.75 is a good starting value.",
            ge=0.01,
            le=5,
            default=0.75,
        ),
        top_p: float = Input(
            description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens",
            ge=0.0,
            le=1.0,
            default=0.9,
        ),
        top_k: int = Input(
            description="When decoding text, samples from the top k most likely tokens; lower to ignore less likely tokens",
            ge=0,
            default=50,
        ),
        use_beam_search: bool = Input(
            description="Whether to use beam search instead of sampling",
            default=False,
        ),
    ) -> ConcatenateIterator:
        """Run a single prediction on the model"""
        promt_formatted = self.tokenizer.apply_chat_template(json.loads(prompt), tokenize=False, add_generation_prompt=True)
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_new_tokens,
            use_beam_search=use_beam_search,
        )
        outputs = self.engine.generate(promt_formatted, sampling_params)
        for output in outputs:
            generated_text = output.outputs[-1].text
            yield generated_text
