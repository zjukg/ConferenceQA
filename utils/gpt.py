import openai
from torch import Tensor
from tqdm.autonotebook import trange
from typing import Optional, Dict, List
from tenacity import retry, wait_random_exponential, stop_after_attempt


def get_api_key(idx: int = 0):
    if idx == 0:
        # your openai api key, for text-embedding-ada-002
        pass 
    elif idx == 1:
        # your openai api key, for gpt
        pass


@retry(wait=wait_random_exponential(min=2, max=10), stop=stop_after_attempt(6))
def get_embedding(text: list | str, model="text-embedding-ada-002") -> list[float]:
    if isinstance(text, str):
        return openai.Embedding.create(input=[text], model=model)["data"][0][
            "embedding"
        ]
    elif isinstance(text, list):
        return openai.Embedding.create(input=text, model=model)


@retry(wait=wait_random_exponential(min=3, max=10), stop=stop_after_attempt(6))
def chat(question: str, gpt: str, temperature=0):
    messages = [{"role": "user", "content": question}]
    resp = openai.ChatCompletion.create(
        model=gpt, messages=messages, temperature=temperature
    )
    return resp.choices[0].message.content


class EncoderAda002:
    def encode(
        self,
        text: List[str],
        batch_size: int = 16,
        show_progress_bar: bool = False,
        **kwargs
    ) -> List[Tensor]:
        text_embeddings = []
        for batch_start in trange(
            0, len(text), batch_size, disable=not show_progress_bar
        ):
            batch_end = batch_start + batch_size
            batch_text = text[batch_start:batch_end]
            # print(f"Batch {batch_start} to {batch_end-1}")
            assert "" not in batch_text
            resp = get_embedding(batch_text)
            for i, be in enumerate(resp["data"]):
                assert (
                    i == be["index"]
                )  # double check embeddings are in same order as input
            batch_text_embeddings = [e["embedding"] for e in resp["data"]]
            text_embeddings.extend(batch_text_embeddings)

        return text_embeddings


class OpenaiAda002:
    def __init__(self) -> None:
        self.q_model = EncoderAda002()
        self.doc_model = self.q_model

    def encode_queries(
        self, queries: List[str], batch_size: int = 16, **kwargs
    ) -> List[Tensor]:
        return self.q_model.encode(queries, batch_size=batch_size, **kwargs)
