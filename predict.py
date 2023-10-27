# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import torch
import json
from cog import BasePredictor, Input, ConcatenateIterator
from tensorizer import TensorDeserializer
from tensorizer.utils import no_init_or_tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, TextIteratorStreamer
from threading import Thread


MODEL_NAME = "teknium/OpenHermes-2-Mistral-7B"
MODEL_CACHE = "model-cache"
TOKEN_CACHE = "token-cache"
CONFIG_CACHE = "config-cache"
TENSORIZED_MODEL_NAME = f"{MODEL_NAME.split('/')[-1]}.tensors"
TENSORIZED_MODEL_PATH = os.path.join(MODEL_CACHE, TENSORIZED_MODEL_NAME)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=TOKEN_CACHE
        )
        config = AutoConfig.from_pretrained(MODEL_NAME, cache_dir=CONFIG_CACHE)
        with no_init_or_tensor():
            self.model = AutoModelForCausalLM.from_config(config)
        deserializer = TensorDeserializer(os.path.join(MODEL_CACHE, TENSORIZED_MODEL_NAME), plaid_mode=True)
        deserializer.load_into_module(self.model)
        deserializer.close()
        self.model.eval()

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
    ) -> ConcatenateIterator:
        """Run a single prediction on the model"""
        messages = self.tokenizer.apply_chat_template(json.loads(prompt), tokenize=False, add_generation_prompt=True)
        tokens_in = self.tokenizer(
            messages,
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids.to('cuda')
        streamer = TextIteratorStreamer(self.tokenizer, timeout=30.0, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(
            input_ids=tokens_in,
            streamer=streamer,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        t = Thread(target=self.model.generate, kwargs=generate_kwargs)
        t.start()
        for out in streamer:
            yield out
