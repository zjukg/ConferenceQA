import os
import logging
import chromadb
import pandas as pd
from typing import List
from chromadb.utils import embedding_functions
from chromadb.api.models.Collection import Collection
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from beir.retrieval import models

from conferenceqa import ConferenceQA
from utils.gpt import OpenaiAda002, get_api_key


logger = logging.getLogger(__name__)


class NewEmbeddingFunction(EmbeddingFunction):
    def __init__(self, encoder) -> None:
        super().__init__()
        self.encoder = encoder

    def __call__(self, texts: Documents) -> Embeddings:
        # embed the documents somehow
        embeddings = self.encoder.encode_queries(texts)
        return embeddings


class Encoder:
    def __init__(self, encoder_name: str) -> None:
        self.encoder_name = encoder_name
        if encoder_name == "text-embedding-002":
            self.encoder = OpenaiAda002()
            self.ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key="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",
                model_name="text-embedding-ada-002",
            )
        elif encoder_name == "SentenceBERT":
            self.encoder = models.SentenceBERT("msmarco-distilbert-base-tas-b")
            self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="msmarco-distilbert-base-tas-b"
            )
        elif encoder_name == "ANCE":
            self.encoder = models.SentenceBERT("msmarco-roberta-base-ance-firstp")
            self.ef = NewEmbeddingFunction(self.encoder)

        self.collection_descs: Collection = None
        self.collection_values: Collection = None
        self.collection_path_and_value: Collection = None

    def _get_embedding_and_save_to_chroma(
        self,
        qa: ConferenceQA = None,
        docs: List[str] = None,
        kind: str = "descs",
        similarity: str = "cosine",
        batch_size: int = 16,
        path: str = None,
    ):
        name = "path" if kind == "descs" else kind
        if not os.path.exists(path):
            print(path)
            chroma_client = chromadb.PersistentClient(path=path)
            logger.info(f"start geting {kind} embeddings...")
            embeddings = self.encoder.doc_model.encode(
                docs, batch_size=batch_size, show_progress_bar=True
            )
            setattr(
                self,
                f"collection_{kind}",
                chroma_client.create_collection(
                    name=name,
                    metadata={"hnsw:space": similarity},
                    embedding_function=self.ef,
                ),
            )
            if not isinstance(embeddings, list):
                embeddings = embeddings.tolist()
            collection = getattr(self, f"collection_{kind}")
            collection.add(
                embeddings=embeddings,
                documents=docs,
                metadatas=[
                    {"source": f"{qa.cfe_name.upper()}2023"} for i in range(len(docs))
                ],
                ids=[str(i) for i in range(len(docs))],
            )
        else:
            chroma_client = chromadb.PersistentClient(path=path)
            setattr(
                self,
                f"collection_{kind}",
                chroma_client.get_collection(
                    name=name,
                    embedding_function=self.ef,
                ),
            )

    def get_embedding(
        self,
        qa: ConferenceQA = None,
        similarity: str = "cosine",
        batch_size: int = 16,
        persist_chroma_path: str = None,
        persist_csv_path: str = None,
    ):
        get_api_key(0)
        self.cfe_name = qa.cfe_name
        descs = qa.descs
        values = qa.values
        paths = qa.paths
        entry_desc = [v for _, v in qa.entry2desc.items()]
        desc_leaf = []
        entry_desc_wo_pre = [v for _, v in qa.entry2desc_wo_pre.items()]

        desc_and_path = [qa.path2desc[path] + f' And the value of this query path is {qa.path2value[path]}' for path in qa.paths]
        path_and_value = [path + ">>" + qa.path2value[path] for path in qa.paths]
        if persist_chroma_path is not None:
            if not os.path.exists(persist_chroma_path):
                os.mkdir(persist_chroma_path)
            descs_path = os.path.join(
                persist_chroma_path,
                f"{self.cfe_name.upper()}2023_{self.encoder_name}_descs",
            )
            self._get_embedding_and_save_to_chroma(
                qa=qa,
                docs=descs,
                kind="descs",
                similarity=similarity,
                batch_size=batch_size,
                path=descs_path,
            )

            paths_path = os.path.join(
                persist_chroma_path, f"{self.cfe_name.upper()}2023_{self.encoder_name}_paths"
            )
            # self._get_embedding_and_save_to_chroma(
            #     qa=qa,
            #     docs=paths,
            #     kind="paths",
            #     similarity=similarity,
            #     batch_size=batch_size,
            #     path=paths_path,
            # )
            
            desc_and_value_path = os.path.join(
                persist_chroma_path, f"{self.cfe_name.upper()}2023_{self.encoder_name}_desc_and_value"
            )
            self._get_embedding_and_save_to_chroma(
                qa=qa,
                docs=desc_and_path,
                kind="desc_and_value",
                similarity=similarity,
                batch_size=batch_size,
                path=desc_and_value_path,
            )

            values_path = os.path.join(
                persist_chroma_path, f"{self.cfe_name.upper()}2023_{self.encoder_name}_values"
            )
            self._get_embedding_and_save_to_chroma(
                qa=qa,
                docs=values,
                kind="values",
                similarity=similarity,
                batch_size=batch_size,
                path=values_path,
            )
        
            path_and_value_path = os.path.join(
                persist_chroma_path,
                f"{self.cfe_name.upper()}2023_{self.encoder_name}_path_and_value",
            )
            
            self._get_embedding_and_save_to_chroma(
                qa=qa,
                docs=path_and_value,
                kind="path_and_value",
                similarity=similarity,
                batch_size=batch_size,
                path=path_and_value_path,
            )
            
            desc_leaf_path = os.path.join(
                persist_chroma_path,
                f"{self.cfe_name.upper()}2023_{self.encoder_name}_desc_leaf",
            )
            self._get_embedding_and_save_to_chroma(
                qa=qa,
                docs=desc_leaf,
                kind="desc_leaf",
                similarity=similarity,
                batch_size=batch_size,
                path=desc_leaf_path,
            )
            
            entry_desc_path = os.path.join(
                persist_chroma_path,
                f"{self.cfe_name.upper()}2023_{self.encoder_name}_entry_desc",
            )
            self._get_embedding_and_save_to_chroma(
                qa=qa,
                docs=entry_desc,
                kind="entry_desc",
                similarity=similarity,
                batch_size=batch_size,
                path=entry_desc_path,
            )

            entry_desc_wo_pre_path = os.path.join(
                persist_chroma_path,
                f"{self.cfe_name.upper()}2023_{self.encoder_name}_entry_desc_wo_pre",
            )
            self._get_embedding_and_save_to_chroma(
                qa=qa,
                docs=entry_desc_wo_pre,
                kind="entry_desc_wo_pre",
                similarity=similarity,
                batch_size=batch_size,
                path=entry_desc_wo_pre_path,
            )
        if persist_csv_path is not None:
            descs_path = os.path.join(
                persist_csv_path,
                f"{self.cfe_name.upper()}2023_{self.encoder_name}_descs.csv",
            )
            if not os.path.exists(descs_path):
                df = pd.DataFrame({"text": descs, "embedding": self.embedding_descs})
                df.to_csv(descs_path, index=False)

            values_path = os.path.join(
                persist_csv_path,
                f"{self.cfe_name.upper()}2023_{self.encoder_name}_values.csv",
            )
            if not os.path.exists(descs_path):
                df = pd.DataFrame({"text": values, "embedding": self.embedding_values})
                df.to_csv(values_path, index=False)
