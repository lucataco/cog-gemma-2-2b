# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, ConcatenateIterator
import os
import time
import torch
import subprocess
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer, Gemma2ForCausalLM
from transformers.generation.streamers import TextIteratorStreamer

# MODEL_URL = "https://weights.replicate.delivery/default/google/gemma-2-2b/model.tar"
MODEL_URL = "https://weights.replicate.delivery/default/google/gemma-2-2b-it/model.tar"
MODEL_CACHE = "checkpoints"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_CACHE, use_fast=True)
        self.model = Gemma2ForCausalLM.from_pretrained(
            MODEL_CACHE,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.model.eval()
        self.executor = ThreadPoolExecutor(max_workers=1)
    
    def predict(
        self,
        prompt: str = Input(
            description="Prompt to send to the model.",
            default="Write me a poem about Machine Learning.",
        ),
        max_new_tokens: int = Input(
            description="Maximum number of tokens to generate. A word is generally 2-3 tokens",
            default=1024, ge=1, le=4096,
        ),
        temperature: float = Input(
            description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic, 0.75 is a good starting value.",
            ge=0.1,
            le=4.0,
            default=0.6,
        ),
        top_p: float = Input(
            description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens",
            ge=0.05,
            le=1.0,
            default=0.9,
        ),
        top_k: int = Input(
            description="When decoding text, samples from the top k most likely tokens; lower to ignore less likely tokens",
            ge=1,
            le=1000,
            default=50,
        ),
        repetition_penalty: float = Input(
            description="Repetition penalty parameter.",
            ge=0.0,
            default=1.2,
        )
    ) -> ConcatenateIterator[str]:
        """Run a single prediction on the model"""
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = {
            **input_ids,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "use_cache": True,
        }
        future = self.executor.submit(self.model.generate, **generate_kwargs)
        try:
            for new_token in streamer:
                yield new_token
        finally:
            future.result()