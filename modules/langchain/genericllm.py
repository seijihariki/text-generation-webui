from langchain.schema import LLMResult, Generation
from langchain.llms.base import BaseLLM

from pydantic import BaseModel
from typing import List, Optional

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
    repetition_penalty: float = 1.15
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

    def _llm_type(self):
        return "text-generation-webui"

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        generations = []
        if stop:
            stop.append('</end>')
        for prompt in prompts:
            prompt_generations = []

            for _ in range(self.generation_attempts):
                cummulative = ''
                for continuation, *_ in generate_reply(prompt,
                                                       self.max_new_tokens,
                                                       self.do_sample,
                                                       self.temperature,
                                                       self.top_p,
                                                       self.typical_p,
                                                       self.repetition_penalty,
                                                       self.encoder_repetition_penalty,
                                                       self.top_k,
                                                       self.min_length,
                                                       self.no_repeat_ngram_size,
                                                       self.num_beams,
                                                       self.penalty_alpha,
                                                       self.length_penalty,
                                                       False,
                                                       self.seed,
                                                       stopping_strings=stop or []):
                    if cummulative == '':
                        cummulative += continuation[1:]
                    else:
                        cummulative += continuation

                    if self.streaming:
                        self.callback_manager.on_llm_new_token(token=continuation)

                    if any(map(lambda x: cummulative.strip().endswith(x), stop or ['</end>'])):
                        break
                prompt_generations.append(Generation(text=cummulative))

            generations.append(prompt_generations)

        return LLMResult(generations=generations)

    async def _agenerate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        return self._generate(prompts=prompts, stop=stop)
