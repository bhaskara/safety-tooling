import asyncio
import logging
import os
import time
import traceback

import openai
import openai._models
import openai.types
from safetytooling.data_models import EmbeddingParams, EmbeddingResponseBase64

from .utils import EMBEDDING_MODELS, price_per_token

LOGGER = logging.getLogger(__name__)


class OpenAIEmbeddingModel:
    def __init__(self, batch_size: int = 2048):
        # TODO: Actually implement a proper rate limit
        self.num_threads = 1
        self.batch_size = batch_size  # Max batch size for embedding endpoint

        self._aclient: openai.AsyncClient | None = None
        self.available_requests = asyncio.BoundedSemaphore(self.num_threads)

    @property
    def aclient(self) -> openai.AsyncClient:
        # The OpenAI client is built on first use: openai>=2 refuses to construct without a key, and
        # InferenceAPI instantiates every provider up front, so an eager client would make the whole
        # API unusable on hosts that carry only Anthropic keys (the cluster, since the 2026-09-04 reset).
        if self._aclient is None:
            if "OPENAI_API_KEY" not in os.environ:
                raise RuntimeError("OpenAI embeddings need OPENAI_API_KEY in the environment")
            self._aclient = openai.AsyncClient()
        return self._aclient

    async def embed(
        self,
        params: EmbeddingParams,
        max_attempts: int = 5,
        **kwargs,
    ) -> EmbeddingResponseBase64:
        assert params.model_id in EMBEDDING_MODELS

        if params.dimensions is not None:
            kwargs["dimensions"] = params.dimensions
        for i in range(max_attempts):
            try:
                async with self.available_requests:
                    # We use base64 format.
                    # See for details https://github.com/EliahKagan/embed-encode/blob/main/why.md
                    response = await self.aclient.embeddings.create(
                        model=params.model_id,
                        input=params.texts,
                        encoding_format="base64",
                        **kwargs,
                    )

                assert all(isinstance(x.embedding, str) for x in response.data)
                total_tokens = response.usage.total_tokens

                return EmbeddingResponseBase64(
                    model_id=response.model,
                    embeddings=[str(x.embedding) for x in response.data],
                    tokens=total_tokens,
                    cost=price_per_token(response.model)[0] * total_tokens,
                )
            except openai.BadRequestError as e:
                raise e
            except Exception as e:
                error_info = (
                    f"Exception Type: {type(e).__name__}, Error Details: {str(e)}, Traceback: {traceback.format_exc()}"
                )
                LOGGER.warn(f"Encountered API error: {error_info}.\nRetrying now. (Attempt {i})")
                time.sleep(1.5**i)

        raise RuntimeError(f"Failed to get a response from the API after {max_attempts} attempts.")
