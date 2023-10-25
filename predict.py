# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "teknium/OpenHermes-2-Mistral-7B"
MODEL_CACHE = "model-cache"
TOKEN_CACHE = "token-cache"

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=TOKEN_CACHE
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            cache_dir=MODEL_CACHE
        ).to('cuda')


    def predict(
        self,
        prompt: str = Input(description="Input prompt not chat template applied"),
        max_new_tokens: int = Input(description="Max new tokens", ge=0, le=1000000000000000019884624838656, default=512),
    ) -> str:
        """Run a single prediction on the model"""
        encodeds = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False
        )
        model_inputs = encodeds.to('cuda')
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True
        )
        decoded = self.tokenizer.batch_decode(generated_ids)
        result = decoded[0]
        return result
