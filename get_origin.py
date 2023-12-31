import json
import argparse
from evaluate import Evaluate
from conferenceqa import ConferenceQA
import tqdm
import os
from utils.api_request_parallel_processor import run
from collections import defaultdict as ddict

tot = []
def evaluate_answer(file_name: str, results, gold):
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
        query = item["query"]
        prompt = """
            You are a judge and need to judge the effect of generated answer.
            
            [task definition]
            Give you two sentences, both of which are answers to the same query, the first sentence is generated by language models, and the second sentence is the reference answer. 
            You need to capture the key information in the reference answer according to the query, and then judge whether the generated answer contains the key information.
            Returns true if the generated answer contains most of the key information, false if the generated answer is wrong. 
            
            
            [note] 
            You only output true or false.
            If the answer contains "As an AI", then we think its answer is wrong.
            As long as it contains key information, it is correct, regardless of whether there is any other uncertain content.
            You should justify your answer, and not let the order of the answers affect it.
        """

        prompt += "\n\n" + "query:" + query
        prompt += "\n\n" + f"generated answer: {d1[query]}"
        prompt += "\n\n" + f"reference answer: {d2[query]}"
        item = {"prompt": prompt, "query": query}
        prompts.append(item)

    resps = run(prompts)
    for resp in resps:
        if "true" in resp["output"].lower():
            tot.append(resp['query'])
    # results.append((num, len(results)))


parser = argparse.ArgumentParser()
parser.add_argument("--cfe_name", default="IJCAI", type=str, help="")
parser.add_argument(
    "--encoder",
    default="text-embedding-002",
    type=str,
    choices=["text-embedding-002", "SentenceBERT", "ANCE"],
    help="",
)
parser.add_argument(
    "--retrieve_method",
    default="desc_leaf",
    type=str,
    choices=[
        "desc_leaf",
        "desc_value",
        "path",
        "path_and_value",
        "desc",
        "desc_and_value",
        "desc_and_path_value",
    ],
    help="",
)
# List[str] path + '>>' + value
parser.add_argument("--dicts_path", default="dataset/IJCAI/dicts", type=str, help="")
parser.add_argument("--embedding_bs", default=200, type=int, help="")
parser.add_argument(
    "--distance",
    default="cosine",
    type=str,
    choices=["cosine", "l2", "ip"],
    help="",
)
parser.add_argument(
    "--persist_chroma_path", default="embeddings/IJCAI", type=str, help=""
)
parser.add_argument("--persist_csv_path", default="embeddings/IJCAI", type=str, help="")
args = parser.parse_args()
qa = ConferenceQA.read_cef(args)

for file_name in [
    "extraction_atomic",
    "extraction_complex",
    "reasoning_atomic",
    "reasoning_complex",
]:
    gold = qa.qas[file_name]
    prompts = []
    for item in gold:
        q = item["question"]
        tmp = {"prompt": f"Answer the question about {args.cfe_name} conference:\n" + q, "query": q}
        prompts.append(tmp)
    res = run(prompts)
    dir = f'/Users/hzw/Desktop/desktop/code/ConferenceQA/dataset/{args.cfe_name}/results'
    with open(os.path.join(dir, f'origin_{file_name}.json'), 'w') as f:
        json.dump(res, f, ensure_ascii=False)
    
