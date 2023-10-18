import os
import json
import tiktoken
import logging
import tqdm
from typing import List, Dict, Tuple
from utils.gpt import get_api_key, get_embedding
from utils.api_request_parallel_processor import run
from utils.cal_sim import cos_sim
import time

logger = logging.getLogger(__name__)


class Retrieve:
    __slot__ = ("qa", "encoder", "retrieve_results")

    def __init__(self, qa, encoder) -> None:
        self.qa = qa
        self.encoder = encoder
        self.retrieve_results = {}

    def get_gpt_answers(
        self, file_name: str, save_path: str = None, prompt_cross: bool = False
    ) -> List[Dict[str, str]]:
        file_path = os.path.join(
            save_path,
            f"{file_name}_{self.encoder.encoder_name}_{self.qa.args.retrieve_method}_answer_results.json",
        )
        results = []
        if save_path is not None:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            # if os.path.exists(file_path):
            #     with open(file_path, "r") as f:
            #         results = json.load(f)
            # else:
            logger.info("strat answer questions...")
            retrieve_results = self.retrieve_results[file_name]
            for query, docs_and_score in retrieve_results.items():
                prompt = "Use the following pieces of context to answer the question. If you don't know the answer, just say that you don't know, don't try to make up an answer."
                if prompt_cross:
                    _new = (
                        list(docs_and_score.keys())[::2]
                        + list(docs_and_score.keys())[1::2]
                    )
                    for s in _new:
                        prompt += "\n\n" + s
                else:
                    for s in docs_and_score.keys():
                        prompt += "\n\n" + s
                prompt += "\n\n" + f"Question:{query}\n" + "Answer:"
                one = {"prompt": prompt, "query": query}
                results.append(one)
            results = run(results)
            with open(
                file_path,
                "w",
            ) as f:
                json.dump(results, f, ensure_ascii=False)
        return results

    def get_retrieve_results(
        self, file_name: str, retrieve_method: str = 'desc', token_num: int = 500, save_path: str = None,  
    ) -> List[Dict[str, Dict]]:
        self.retrieve_results[file_name] = {}
        file_path = os.path.join(
            save_path,
            f"{file_name}_{self.encoder.encoder_name}_{self.qa.args.retrieve_method}_retrieve_results.json",
        )
        if save_path is not None:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            # if os.path.exists(file_path):
            #     with open(file_path, "r") as f:
            #         self.retrieve_results[file_name] = json.load(f)
            # else:
            logger.info("start retrieve...")
            for item in tqdm.tqdm(self.qa.qas[file_name], total=len(self.qa.qas[file_name])):
                retrieve_func = getattr(self, f'retrieve_{retrieve_method}')
                topk = retrieve_func(item["question"], token_num=token_num)
                # result = {"question": item["question"], "docs_and_score": topk}
                # self.retrieve_results[file_name].append(result)
                self.retrieve_results[file_name][item["question"]] = topk
                time.sleep(2)

            with open(
                file_path,
                "w",
            ) as f:
                json.dump(self.retrieve_results[file_name], f, ensure_ascii=False)

        return self.retrieve_results[file_name]

    def retrieve_desc_value(self, query: str, token_num: int = 500) -> Dict[str, float]:
        get_api_key(0)
        collection = self.encoder.collection_desc_and_value
        results = collection.query(query_texts=[query], n_results=500)
        dis_desc_and_value, ret_desc_and_value = results["distances"][0], results["documents"][0]
        path2score = {self.qa.desc2path[k.split(' And the value of this query path is ')[0]]: v for k, v in zip(ret_desc_and_value, dis_desc_and_value)}
        res = set()
        for k, _ in path2score.items():
            res.add(k)
        score = [(path2score.get(i, 0), i + ">>" + self.qa.path2value[i]) for i in res]
        score.sort()
        # final = [idx[1] for idx in score]

        encoding = tiktoken.get_encoding("cl100k_base")
        total = 0
        for idx, (_, doc) in enumerate(score):
            tokens_len = len(encoding.encode(doc))
            if total + tokens_len < token_num:
                total += tokens_len
            else:
                score = score[:idx]
                break
        final = {v: k for k, v in score}
        return final
    
    def retrieve_path(self, query: str, token_num: int = 500) -> Dict[str, float]:
        get_api_key(0)
        collection = self.encoder.collection_paths
        results = collection.query(query_texts=[query], n_results=500)
        dis_path, ret_path = results["distances"][0], results["documents"][0]
        path2score = {k: v for k, v in zip(ret_path, dis_path)}
        res = set()
        for k, _ in path2score.items():
            res.add(k)
        score = [(path2score.get(i, 0), i + ">>" + self.qa.path2value[i]) for i in res]
        score.sort()
        # final = [idx[1] for idx in score]

        encoding = tiktoken.get_encoding("cl100k_base")
        total = 0
        for idx, (_, doc) in enumerate(score):
            tokens_len = len(encoding.encode(doc))
            if total + tokens_len < token_num:
                total += tokens_len
            else:
                score = score[:idx]
                break
        final = {v: k for k, v in score}
        return final
    
    def retrieve_desc_leaf(self, query: str, token_num: int = 500) -> Dict[str, float]:
        get_api_key(0)
        collection = self.encoder.collection_desc_leaf
        # dis_desc, ret_desc = Retrieve.get_retrieve_results_wo_chroma(query=query, collection=collection)
        results = collection.query(query_texts=[query], n_results=500)
        dis_desc, ret_desc = results["distances"][0], results["documents"][0]
        path2score = {self.qa.desc2pathvalue[k]: v for k, v in zip(ret_desc, dis_desc)}
        res = set()
        for k, _ in path2score.items():
            res.add(k)
        score = [(path2score.get(i, 0), i) for i in res]
        score.sort()
        # final = [idx[1] for idx in score]

        encoding = tiktoken.get_encoding("cl100k_base")
        total = 0
        for idx, (_, doc) in enumerate(score):
            tokens_len = len(encoding.encode(doc))
            if total + tokens_len < token_num:
                total += tokens_len
            else:
                score = score[:idx]
                break
        final = {v: k for k, v in score}
        return final
    
    def retrieve_desc(self, query: str, token_num: int = 500) -> Dict[str, float]:
        get_api_key(0)
        collection = self.encoder.collection_descs
        results = collection.query(query_texts=[query], n_results=500)
        dis_desc, ret_desc = results["distances"][0], results["documents"][0]
        path2score = {self.qa.desc2path[k]: v for k, v in zip(ret_desc, dis_desc)}
        res = set()
        for k, _ in path2score.items():
            res.add(k)
        score = [(path2score.get(i, 0), i + ">>" + self.qa.path2value[i]) for i in res]
        score.sort()
        # final = [idx[1] for idx in score]

        encoding = tiktoken.get_encoding("cl100k_base")
        total = 0
        for idx, (_, doc) in enumerate(score):
            tokens_len = len(encoding.encode(doc))
            if total + tokens_len < token_num:
                total += tokens_len
            else:
                score = score[:idx]
                break
        final = {v: k for k, v in score}
        return final
    
    def retrieve_entry_desc(self, query: str, token_num: int = 500) -> Dict[str, float]:
        get_api_key(0)
        collection = self.encoder.collection_entry_desc
        results = collection.query(query_texts=[query], n_results=500)
        dis_desc, ret_desc = results["distances"][0], results["documents"][0]
        entry2score = {self.qa.desc2entry[k]: v for k, v in zip(ret_desc, dis_desc)}
        res = set()
        for k, _ in entry2score.items():
            res.add(k)
        score = [(entry2score.get(i, 0), i) for i in res]
        score.sort()
        # final = [idx[1] for idx in score]

        encoding = tiktoken.get_encoding("cl100k_base")
        total = 0
        for idx, (_, doc) in enumerate(score):
            tokens_len = len(encoding.encode(doc))
            if total + tokens_len < token_num:
                total += tokens_len
            else:
                score = score[:idx]
                break
        final = {v: k for k, v in score}
        return final

    def retrieve_entry_desc_wo_pre(self, query: str, token_num: int = 500) -> Dict[str, float]:
        get_api_key(0)
        collection = self.encoder.collection_entry_desc_wo_pre
        results = collection.query(query_texts=[query], n_results=500)
        dis_desc, ret_desc = results["distances"][0], results["documents"][0]
        entry2score = {self.qa.desc_wo_pre2entry[k]: v for k, v in zip(ret_desc, dis_desc)}
        res = set()
        for k, _ in entry2score.items():
            res.add(k)
        score = [(entry2score.get(i, 0), i) for i in res]
        score.sort()
        # final = [idx[1] for idx in score]

        encoding = tiktoken.get_encoding("cl100k_base")
        total = 0
        for idx, (_, doc) in enumerate(score):
            tokens_len = len(encoding.encode(doc))
            if total + tokens_len < token_num:
                total += tokens_len
            else:
                score = score[:idx]
                break
        final = {v: k for k, v in score}
        return final
    
    def retrieve_desc_and_path_value(
        self, query: str, token_num: int = 500
    ) -> Dict[str, float]:
        get_api_key(0)
        assert len(self.qa.descs) == len(self.qa.path_and_value)
        collection_descs = self.encoder.collection_descs
        collection_path_and_value = self.encoder.path_and_value
        results_descs = collection_descs.query(
            query_texts=[query], n_results=len(self.qa.descs)
        )
        results_path_and_value = collection_path_and_value.query(
            query_texts=[query], n_results=len(self.qa.path2value)
        )
        dis_desc, ret_desc = (
            results_descs["distances"][0],
            results_descs["documents"][0],
        )
        dis_path_and_value, ret_path_and_value = (
            results_path_and_value["distances"][0],
            results_path_and_value["documents"][0],
        )
        path2score = {self.qa.desc2path[k]: v for k, v in zip(ret_desc, dis_desc)}
        path_and_value2score = {
            k: v for k, v in zip(ret_path_and_value, dis_path_and_value)
        }
        score = [
            (
                (path2score[i] * 0.7 + path_and_value2score[i + ">>" + self.qa.path2value[i]] * 0.3),
                i + ">>" + self.qa.path2value[i],
            )
            for i in set([k for k, _ in path2score.items()])
        ]
        score.sort()

        encoding = tiktoken.get_encoding("cl100k_base")
        total = 0
        for idx, value in enumerate(final):
            tokens_len = len(encoding.encode(value))
            if total + tokens_len < token_num:
                total += tokens_len
            else:
                score = score[:idx]
                break
        
        final = {v: k for k, v in score}
        return final

    def retrieve_desc_and_value(self, query: str, token_num: int = 500) -> Dict[str, float]:
        get_api_key(0)
        assert len(self.qa.descs) == len(self.qa.values)
        collection_descs = self.encoder.collection_descs
        collection_values = self.encoder.collection_values
        results_descs = collection_descs.query(
            query_texts=[query], n_results=500
        )
        results_values = collection_values.query(
            query_texts=[query], n_results=500
        )
        dis_desc, ret_desc = (
            results_descs["distances"][0],
            results_descs["documents"][0],
        )
        dis_value, ret_value = (
            results_values["distances"][0],
            results_values["documents"][0],
        )
        path2score = {self.qa.desc2path[k]: v for k, v in zip(ret_desc, dis_desc)}
        value2score = {k: v for k, v in zip(ret_value, dis_value)}
        score = [
            (
                # self.qa.args.lab * path2score[i] + (1 - self.qa.args.lab) * value2score[self.qa.path2value[i]],
                self.qa.args.lab * path2score.get(i, 100) + (1 - self.qa.args.lab) * value2score.get(self.qa.path2value[i], 100),
                i + ">>" + self.qa.path2value[i],
            )
            for i in set([k for k, _ in path2score.items()])
        ]
        score.sort()

        encoding = tiktoken.get_encoding("cl100k_base")
        total = 0
        for idx, (_, doc) in enumerate(score):
            tokens_len = len(encoding.encode(doc))
            if total + tokens_len < token_num:
                total += tokens_len
            else:
                score = score[:idx]
                break
            
        final = {v: k for k, v in score}
        return final

    def retrieve_path_and_value(self, query: str, token_num: int = 500) -> Dict[str, float]:
        get_api_key(0)
        collection = self.encoder.collection_path_and_value
        results = collection.query(query_texts=[query], n_results=500)
        dis_path_and_value, ret_path_and_value = results["distances"][0], results["documents"][0]
        path_and_value2score = {k: v for k, v in zip(ret_path_and_value, dis_path_and_value)}
        res = set()
        for k, _ in path_and_value2score.items():
            res.add(k)
        score = [(path_and_value2score.get(i, 0), i) for i in res]
        score.sort()
        # final = [idx[1] for idx in score]

        encoding = tiktoken.get_encoding("cl100k_base")
        total = 0
        for idx, (_, doc) in enumerate(score):
            tokens_len = len(encoding.encode(doc))
            if total + tokens_len < token_num:
                total += tokens_len
            else:
                score = score[:idx]
                break
        final = {v: k for k, v in score}
        return final
    
    @staticmethod
    def get_retrieve_results_wo_chroma(query: str, collection) -> Tuple[List[str], List[float]]:
        q = get_embedding(query)
        tmp = collection.get(include=['documents', 'embeddings'])
        docs = tmp['documents']
        embs = tmp['embeddings']
        doc_emb = [(1.0 - cos_sim(q, e), d, e) for d, e in zip(docs, embs)]
        doc_emb.sort()
        
        dis, doc, _ = zip(*doc_emb)
        
        return dis[:500], doc[:500]
        
        
        