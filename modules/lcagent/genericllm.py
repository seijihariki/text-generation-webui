from langchain.schema import LLMResult, Generation
from langchain.llms.base import BaseLLM

from pydantic import BaseModel
from typing import Any, List, Optional

from modules.text_generation import generate_reply


class WebUILLM(BaseLLM, BaseModel):
    streaming: bool = False
    """Whether to stream the results or not."""
    generation_attempts: int = 1
    """Number of generations per prompt."""
    max_new_tokens: int = 200
    """Maximum number of newly generated tokens."""
    do_sample: bool = True
    """Whether to do sample."""
    temperature: float = 0.72
    """Creativity of the model."""
    top_p: float = 0.18
    """Top P."""
    typical_p: float = 1
    """Typical P."""
    top_k: int = 30
    """Top K."""
    min_length: int = 0
    """Minimum length of the generated result."""
    repetition_penalty: float = 1.5
    """Penalizes repetition."""
    encoder_repetition_penalty: float = 1
    """Penalizes encoder repetition."""
    penalty_alpha: float = 0
    """Alpha for Contrastive Search penalties."""
    no_repeat_ngram_size: int = 0
    """Size of ngrams for repetition penalty."""
    num_beams: int = 1
    """Number of beams."""
    length_penalty: int = 1
    """Penalizes length."""
    seed: int = -1
    """Generation Seed."""

    generate_state: Any = None

    def _llm_type(self):
        return "text-generation-webui"

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = []) -> LLMResult:
        generations = []
        self.generate_state['max_new_tokens'] = self.max_new_tokens

        if not stop:
            stop = []
        stop.append('</end>')

        with open('llm-file.log', 'w') as f:
            f.write(prompts[0])

        for prompt in prompts:
            prompt_generations = []
            prompt_length = len(prompt)

            for _ in range(self.generation_attempts):
                generated_length = 0
                generated_string = ""
                for continuation in generate_reply(prompt,
                                                       self.generate_state,
                                                       stopping_strings=stop):
                    old_generated_length = generated_length
                    generated_length = len(continuation) - prompt_length

                    continuation = continuation[prompt_length + old_generated_length:]
                    generated_string += continuation

                    if self.streaming:
                        self.callback_manager.on_llm_new_token(token=continuation)

                    if any(map(lambda x: generated_string.strip().endswith(x), stop)):
                        break
                prompt_generations.append(Generation(text=generated_string))

            generations.append(prompt_generations)

        print('G:', generations, flush=True)
        return LLMResult(generations=generations)

    async def _agenerate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        return self._generate(prompts=prompts, stop=stop)
