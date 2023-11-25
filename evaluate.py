import tqdm
import logging
import pytrec_eval
from typing import Dict, List, Tuple
from utils.api_request_parallel_processor import run
import string
import collections
import re
import json

logger = logging.getLogger(__name__)

def _normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def F1_compute(answer, pred):
    def get_tokens(s):
        if not s: return []
        return _normalize_answer(s).split()

    def compute_f1(a_gold, a_pred):
        gold_toks = get_tokens(a_gold)
        pred_toks = get_tokens(a_pred)
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
    
    return compute_f1(answer, pred.split('###')[0])


class Evaluate:
    @staticmethod
    def evaluate(
        qrels: Dict[str, Dict[str, int]],
        results: Dict[str, Dict[str, float]], 
        k_values: List[int],
        ignore_identical_ids: bool = False,
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
        if ignore_identical_ids:
            logger.info(
                "For evaluation, we ignore identical query and document ids (default), please explicitly set ``ignore_identical_ids=False`` to ignore this."
            )
            popped = []
            for qid, rels in results.items():
                for pid in list(rels):
                    if qid == pid:
                        results[qid].pop(pid)
                        popped.append(pid)

        results = {
            query: {doc: 1 - score for doc, score in results[query].items()}
            for query, _ in results.items()
        }

        ndcg = {}
        _map = {}
        recall = {}
        precision = {}

        for k in k_values:
            ndcg[f"NDCG@{k}"] = 0.0
            _map[f"MAP@{k}"] = 0.0
            recall[f"Recall@{k}"] = 0.0
            precision[f"P@{k}"] = 0.0

        map_string = "map_cut." + ",".join([str(k) for k in k_values])
        ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
        recall_string = "recall." + ",".join([str(k) for k in k_values])
        precision_string = "P." + ",".join([str(k) for k in k_values])
        evaluator = pytrec_eval.RelevanceEvaluator(
            qrels, {map_string, ndcg_string, recall_string, precision_string}
        )
        scores = evaluator.evaluate(results)

        for query_id in scores.keys():
            for k in k_values:
                ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
                _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
                recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
                precision[f"P@{k}"] += scores[query_id]["P_" + str(k)]

        for k in k_values:
            ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(scores), 5)
            _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / len(scores), 5)
            recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / len(scores), 5)
            precision[f"P@{k}"] = round(precision[f"P@{k}"] / len(scores), 5)

        with open("result.txt", "a") as file:
            for eval in [ndcg, _map, recall, precision]:
                for k in eval.keys():
                    file.write("{}: {:.4f}".format(k, eval[k]) + '   ')
                file.write("\n")

        return ndcg, _map, recall, precision

    @staticmethod
    def evaluate_answer(
        file_name: str,
        results: List[Dict[str, str]],
        gold: List[Dict[str, str]],
        exp_name: str = "",
    ):
        prompts = []
        d1 = {}
        for item in results:
            try:
                d1[item["question"]] = item["answer"]
            except:
                d1[item["query"]] = item["output"]

        d2 = {}
        for item in gold:
            try:
                d2[item["question"]] = item["answer"]
            except:
                d2[item["query"]] = item["output"]
        num = 0
        for step, item in tqdm.tqdm(enumerate(results), total=len(results)):
            try:
                query = item["query"]
            except:
                query = item["question"]
            prompt = """
                You are a judge and need to judge the effect of generated answer.
                
                [task definition]
                Give you two sentences, both of which are answers to the same query, the first sentence is generated by language models, and the second sentence is the reference answer. 
                You need to capture the key information in the reference answer according to the query, and then judge whether the generated answer contains the key information.
                Returns true if the generated answer contains most of the key information, false if the generated answer is wrong. 
                As long as it contains key information, it is correct, regardless of whether there is any other uncertain content.
                
                [note] 
                You only output true or false.
                You should justify your answer, and not let the order of the answers affect it.
            """

            prompt += "\n\n" + "query:" + query
            prompt += "\n\n" + f"generated answer: {d1[query]}"
            prompt += "\n\n" + f"reference answer: {d2[query]}"
            item = {"prompt": prompt, "query": ""}
            prompts.append(item)

        resps = run(prompts)
        for resp in resps:
            if "true" in resp["output"].lower():
                num += 1
        with open("result.txt", "a") as file:
            file.write(f"{exp_name}: {file_name}: ({num}/{len(results)})" + "\n")
        logger.info(f"{file_name}: ({num}/{len(results)})")
        
    @staticmethod
    def evaluate_f1(
        file_name: str,
        results: List[Dict[str, str]],
        gold: List[Dict[str, str]],
        exp_name: str = "",
    ):
        prompts = []
        d1 = {}
        for item in results:
            try:
                d1[item["question"]] = item["answer"]
            except:
                d1[item["query"]] = item["output"]

        d2 = {}
        for item in gold:
            try:
                d2[item["question"]] = item["answer"]
            except:
                d2[item["query"]] = item["output"]
        score = 0
        for step, item in tqdm.tqdm(enumerate(results), total=len(results)):
            try:
                query = item["query"]
            except:
                query = item['question']
            gold = d1[query]
            pred = d2[query]
            score += F1_compute(gold, pred) 
        res = round(score/len(results), 4)
        with open("result.txt", "a") as file:
            file.write(f"{exp_name}: {file_name}: {res}" + "\n")
        logger.info(f"{file_name}: {res}")