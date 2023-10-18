import json
import logging
import os
import argparse
from encoder import Encoder
from retrieve import Retrieve
from evaluate import Evaluate
from conferenceqa import ConferenceQA
from utils.logging import LoggingHandler
from typing import List, Dict

def set_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
    )

    file_handler = logging.FileHandler("log/1.log", mode="a")
    file_handler.setLevel(level=logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[LoggingHandler(), stream_handler],
    )

    # return logger


if __name__ == "__main__":
    set_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfe_name", default="WWW", type=str, help="")
    parser.add_argument(
        "--encoder",
        default="text-embedding-002",
        type=str,
        choices=["text-embedding-002", "SentenceBERT", 'ANCE'],
        help="",
    )
    parser.add_argument(
        "--retrieve_method",
        default="entry_desc_wo_pre",
        type=str,
        choices=["entry_desc_wo_pre","entry_desc", "desc_leaf", "desc_value" ,"path", "path_and_value", "desc", "desc_and_value", 'desc_and_path_value'],
        help="",
    )
    # List[str] path + '>>' + value
    parser.add_argument("--dicts_path", default="dataset/WWW/dicts", type=str, help="")
    parser.add_argument("--embedding_bs", default=50, type=int, help="")
    parser.add_argument(
        "--distance",
        default="cosine",
        type=str,
        choices=["cosine", "l2", "ip"],
        help="",
    )
    parser.add_argument(
        "--persist_chroma_path", default="embeddings/WWW", type=str, help=""
    )
    parser.add_argument(
        "--persist_csv_path", default="embeddings/WWW", type=str, help=""
    )
    parser.add_argument("--lab", default=0, type=float, help="")
    args = parser.parse_args()
    qa = ConferenceQA.read_cef(args)
    qrels = qa.qrels
    
    encoder = Encoder(args.encoder)
    encoder.get_embedding(
        qa,
        args.distance,
        args.embedding_bs,
        args.persist_chroma_path,
        # args.persist_csv_path
    )

    retriever = Retrieve(qa, encoder)
    
    exp_name = f'{args.cfe_name}|{args.encoder}|{args.distance}|{args.retrieve_method}|{args.lab}'
    with open("result.txt", "a") as file:
        file.write('\n\n')
        file.write('*'*10 + f'{exp_name}' + '*'*10 + '\n')
    for file_name in [
        "extraction_atomic",
        "extraction_complex",
        "reasoning_atomic",
        "reasoning_complex",
    ]:
        retrieve_results: List[Dict[str, Dict]] = retriever.get_retrieve_results(
            file_name=file_name, retrieve_method=args.retrieve_method, save_path=f"dataset/{args.cfe_name.upper()}/results"
        )  # 没有save_path则不保存，否则覆盖保存

        qrels = qa.qrels

        Evaluate.evaluate(qrels=qrels, results=retrieve_results, k_values=[1, 5])

        answer_results: List[Dict[str, str]] = retriever.get_gpt_answers(
            file_name=file_name, save_path=f"dataset/{args.cfe_name.upper()}/results"
        )
        
        gold = qa.qas[file_name]
        Evaluate.evaluate_answer(file_name=file_name, results=answer_results, gold=gold, exp_name=exp_name)
        Evaluate.evaluate_f1(file_name=file_name, results=answer_results, gold=gold, exp_name=exp_name)