from rank_bm25 import BM25Okapi
import json
import tqdm
import time
import os
import re as rem
import openai
from typing import List

def flatten_json(json_data, prefix='', result=None):
    if result is None:
        result = []

    for key in json_data:
        if isinstance(json_data[key], dict):
            flatten_json(json_data[key], prefix + key + ': ', result)
        elif isinstance(json_data[key], list):
            for li in json_data[key]:
                if isinstance(li, dict):
                    flatten_json(li, prefix + key + ': ', result)
                elif isinstance(li, list):
                    if isinstance(li[0], dict):
                        for lli in li:
                            flatten_json(lli, prefix + key + ': ', result)
                    else:
                        for lli in li:
                           result.append(prefix + key + ': ' + lli + ',')     
                else:
                    result.append(prefix + key + ': ' + li + ',')
        else:
            result.append(prefix + key + ': ' + str(json_data[key]) + '.')

    return result


def encode(text: str)-> List[str]:
    matches = rem.findall(r'\d+\.\s*+(.*)', text)
    print(matches)
    return matches
    if matches:
        for match in matches:
            print(match)
    
conf = 'IJCAI'
dir = f'/Users/hzw/Desktop/desktop/code/ConferenceQA/dataset/{conf}'
with open(os.path.join(dir, f'{conf}2023.json'),
          "r",
        #   encoding='utf-8-sig'
          ) as f:
    json_data = json.load(f)

# 将 JSON 数据平铺并转化为字符串
corpus = flatten_json(json_data)

tokenized_corpus = [doc.split(" ") for doc in corpus]

bm25 = BM25Okapi(tokenized_corpus)

pres = []
recs = []
time_all = []
def get_result(file_name):
    with open(os.path.join(dir, f'{file_name}.json'), 'r') as file:
        json_data = file.read()

    try:
        data_dict = json.loads(json_data)['QAs']
    except: 
        data_dict = json.loads(json_data)
        
    results = []
    re = {}
    for step, item in tqdm.tqdm(enumerate(data_dict), total=len(data_dict)):
        query = item['question']
        pattern = rf'{conf}2023/.*?(?={conf}2023/|$)'
        refs = rem.findall(pattern, item['from'])
        if len(refs) == 0:
            refs = [item['from']]
        tokenized_query = query.split(" ")
        re[str(step)] = bm25.get_top_n(tokenized_query, corpus, n=5)
        
        prompt = "\nList the text that you find helpful in answering the question, return it in the format (Helpful references:\n1. \n2. \n3.)"
        for s in re[str(step)]:
            prompt += "\n\n" + s
        prompt += "\n\n" + f"Question:{query}" + "Helpful References:"
        
        num = 0
        while True:
            try:
                response = openai.ChatCompletion.create(
                    deployment_id="gpt35",
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "user", 
                            "content": prompt
                        }]
                )
                break
            except:
                num += 1
                print(f'sleep {2.5*num}')
                time.sleep(2.5*num)
        res = {
                'question': query,
                'answer': response.choices[0].message.content
            }
        results.append(res)
    
    with open(os.path.join(dir, 'results', f'bm25_{file_name}.json'), 'w') as file:
        json.dump(results, file)

for i, file_name in enumerate(['extraction_atomic','extraction_complex','reasoning_atomic','reasoning_complex']):
    get_result(file_name)

    