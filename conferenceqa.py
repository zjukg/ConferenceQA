import os
import re
import json
import time
import tqdm
import random
import openai
import logging
import chromadb

from collections import defaultdict as ddcit
from chromadb.utils import embedding_functions
from collections import defaultdict as ddict
from typing import Optional, Union, Sequence, List, Dict, Any
from tenacity import retry, wait_random_exponential, stop_after_attempt

from utils.gpt import get_api_key
from utils.api_request_parallel_processor import run

logger = logging.getLogger(__name__)


class Node:
    def __init__(self, name: str, is_leaf: bool = 0):
        self.name = name
        self.children = []
        self.parent = None
        self.is_leaf = is_leaf

    def add_child(self, child):
        self.children.append(child)
        child.parent = self


class ConferenceQA:
    def __init__(self, args, cfe_json_data: Dict, root: Node) -> None:
        self.args = args
        self.cfe_name = args.cfe_name
        self.cfe_json_data = cfe_json_data
        self.root = root
        self.dicts_path = args.dicts_path
        if not os.path.exists(self.dicts_path):
            os.mkdir(self.dicts_path)

        logger.info("loading path2value...")
        if os.path.exists(os.path.join(self.dicts_path, "path2value.json")):
            with open(os.path.join(self.dicts_path, "path2value.json"), "r") as f:
                self.path2value = json.load(f)
        else:
            self.path2value = {}
            self._bfs_traverse(self.root)
            self.dict_dump(
                self.path2value, os.path.join(self.dicts_path, "path2value.json")
            )

        self.value2path = {v: k for k, v in self.path2value.items()}
        self.values = [v for _, v in self.path2value.items()]
        self.paths = [k for k, _ in self.path2value.items()]

        logger.info("loading path2desc...")
        if os.path.exists(os.path.join(self.dicts_path, "path2desc.json")):
            with open(os.path.join(self.dicts_path, "path2desc.json"), "r") as f:
                self.path2desc = json.load(f)
        else:
            self.path2desc = {}
            self.get_structure_description(
                self.root,
                x2desc=self.path2desc,
                file_path=os.path.join(self.dicts_path, "path2desc.json"),
            )
            self.dict_dump(
                self.path2desc, os.path.join(self.dicts_path, "path2desc.json")
            )

        self.desc2path = {v: k for k, v in self.path2desc.items()}
        self.descs = [self.path2desc[k] for k, _ in self.path2value.items()]

        logger.info("loading entry2desc...")
        if os.path.exists(os.path.join(self.dicts_path, "entry2desc.json")):
            with open(os.path.join(self.dicts_path, "entry2desc.json"), "r") as f:
                self.entry2desc = json.load(f)
        else:
            self.entry2desc = {}
            entries = [k + ">>" + v for k, v in self.path2value.items()]
            self.get_description(
                entries
                # x2desc=self.entry2desc,
                # file_path=os.path.join(self.dicts_path, "entry2desc.json"),
            )
            self.dict_dump(
                self.entry2desc, os.path.join(self.dicts_path, "entry2desc.json")
            )
            print(self.entry2desc)
        self.desc2entry = {v: k for k, v in self.entry2desc.items()}

        logger.info("loading entry2desc_wo_pre...")
        if os.path.exists(os.path.join(self.dicts_path, "entry2desc_wo_pre.json")):
            with open(os.path.join(self.dicts_path, "entry2desc_wo_pre.json"), "r") as f:
                self.entry2desc_wo_pre = json.load(f)
        else:
            # with open(os.path.join(self.dicts_path, "entry2desc_wo_pre.json"), "r") as f:
            #     self.entry2desc_wo_pre = json.load(f)
            # print(len(self.pathvalue2desc.keys()))
            # import pdb;pdb.set_trace()
            # assert 'WWW2023>>Home>>keynotes>>keynotes_5' in self.pathvalue2desc.keys()
            self.entry2desc_wo_pre = {}
            self.get_structure_description(
                self.root,
                x2desc=self.entry2desc_wo_pre,
                file_path=os.path.join(self.dicts_path, "entry2desc_wo_pre.json"),
                leaf=True,
            )
            self.dict_dump(
                self.entry2desc_wo_pre,
                os.path.join(self.dicts_path, "entry2desc_wo_pre.json"),
            )

        self.desc_wo_pre2entry = {v: k for k, v in self.entry2desc_wo_pre.items()}
        
        logger.info("loading pathvalue2desc...")
        if os.path.exists(os.path.join(self.dicts_path, "pathvalue2desc.json")):
            with open(os.path.join(self.dicts_path, "pathvalue2desc.json"), "r") as f:
                self.pathvalue2desc = json.load(f)
        else:
            with open(os.path.join(self.dicts_path, "path2desc.json"), "r") as f:
                self.pathvalue2desc = json.load(f)
            # import pdb;pdb.set_trace()
            self.get_structure_description(
                self.root,
                x2desc=self.pathvalue2desc,
                file_path=os.path.join(self.dicts_path, "pathvalue2desc.json"),
                leaf=True,
            )
            self.dict_dump(
                self.pathvalue2desc,
                os.path.join(self.dicts_path, "pathvalue2desc.json"),
            )

        self.desc2pathvalue = {v: k for k, v in self.pathvalue2desc.items()}
        self.path_and_value = [
            path + ">>" + self.path2value[path] for path in self.paths
        ]

        self.qas = {}
        self.qrels = {}
        for file_name in [
            "extraction_atomic",
            "extraction_complex",
            "reasoning_atomic",
            "reasoning_complex",
        ]:
            try:
                data_dict = json.load(
                    open(f"dataset/{self.cfe_name.upper()}/{file_name}.json", "r")
                )["QAs"]
            except:
                data_dict = json.load(
                    open(f"dataset/{self.cfe_name.upper()}/{file_name}.json", "r")
                )
            self.qas[file_name] = data_dict
            for item in data_dict:
                self.qrels[item["question"]] = self.modify_from(item["from"])

    def modify_from(self, text: str | list = None):
        if isinstance(text, list):
            refs = text
        else:
            refs = []
            if ">>" in text:
                refs = [text]
            else:
                pattern = rf"{self.cfe_name.upper()}2023/.*?(?={self.cfe_name.upper()}2023/|$)"
                refs = re.findall(pattern, text)
                for i, ref in enumerate(refs):
                    ref = ref.strip()
                    if ref[-1] == ",":
                        ref = ref[:-1]
                    ref = ref.replace("/", ">>")
                    refs[i] = ref
        doc_rel = {}
        for doc in [path + ">>" + self.path2value[path] for path in self.paths]:
            if any(ref in doc for ref in refs):
                doc_rel[doc] = 1
            else:
                doc_rel[doc] = 0
        return doc_rel

    def dict_dump(self, data: Dict | List, file_path: str):
        with open(file_path, "w") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def read_cef(cls, args) -> "ConferenceQA":
        try:
            with open(
                f"dataset/{args.cfe_name.upper()}/{args.cfe_name.upper()}2023.json",
                "r",
            ) as f:
                cfe_json_data = json.load(f)
        except:
            with open(
                f"dataset/{args.cfe_name.upper()}/{args.cfe_name.upper()}2023.json",
                "r",
                encoding="utf-8-sig",
            ) as f:
                cfe_json_data = json.load(f)
        root = Node(f"{args.cfe_name.upper()}2023")
        root = cls._build(root=root, data=cfe_json_data[f"{args.cfe_name.upper()}2023"])
        return cls(args=args, cfe_json_data=cfe_json_data, root=root)

    @classmethod
    def _build(cls, root: Node, data: Node) -> Node:
        if isinstance(data, str):
            root.add_child(Node(str(data), 1))

        elif isinstance(data, list):
            for i, v in enumerate(data):
                child = Node(root.name + f"_{i+1}")
                root.add_child(child)
                cls._build(child, v)
        elif isinstance(data, dict):
            for key, value in data.items():
                node = Node(key)
                root.add_child(node)
                if isinstance(value, dict):
                    cls._build(node, value)
                elif isinstance(value, list):
                    for i, v in enumerate(value):
                        child = Node(key + f"_{i+1}")
                        node.add_child(child)
                        cls._build(child, v)
                elif isinstance(value, str):
                    node.add_child(Node(str(value), 1))
        else:
            raise TypeError
        return root

    def _bfs_traverse(self, root: Node) -> None:
        if not root:
            return
        queue = [root]
        while queue:
            paths = []
            l = len(queue)
            for i in range(l):
                node = queue.pop(0)
                u = node
                path = [u.name]
                while u.parent is not None:
                    u = u.parent
                    path.append(u.name)
                path.reverse()

                paths.append(">>".join(path))
                for child in node.children:
                    if not child.is_leaf:
                        queue.append(child)
                    else:
                        if child.name == "":
                            child.name = "unkonwn"
                        self.path2value[">>".join(path)] = child.name
                        assert child.name != ""

    def get_description(self, entries: List[str]):
        print(len(entries))
        time.sleep(5)
        prompts = []
        for entry in entries:
            prompt = """
            Convert [Path] into sentences describing it.
            
            [Path] 
            ##1##
            """
            prompt = prompt.replace('##1##', entry)
            item = {"prompt": prompt, "query": entry}
            prompts.append(item)
        results = run(prompts)
        for result in results:
            self.entry2desc[result["query"]] = result["output"]

    def get_structure_description(
        self, root: Node, x2desc: Dict, file_path: str, leaf: bool = False
    ):
        if not root:
            return
        all_leaf = []
        prompts = []
        tmp = []
        queue = [root]
        while queue:
            prompts = []
            paths = []
            l = len(queue)
            for i in range(l):
                node = queue.pop(0)
                prompt_root = """
                turn the [Query Path] to text based on [Requirements] and [Relevant Information]. 
                
                [Requirements]
                The rewritten text should include all the relevant information of its surrounding nodes for easy retrieval.

                [Relevant Information] 
                All or part of sub-level nodes(no more than 10): ##2## 

                [Query Path] 
                ##5##
                """

                prompt = """
                turn the [Query Path] to text based on [Requirements] and [Relevant Information]. 
                
                [Requirements]
                The rewritten text should include all the relevant information of its surrounding nodes for easy retrieval.

                [Relevant Information] 
                All or part of same level nodes(no more than 10): ##1## 
                Tip: The meaning of ##3##: ##4## 

                [Query Path] 
                ##5##
                """

                # ablation study
                prompt_leaf = """
                Turn the [Query Path] to text based on[Relevant Information].
                
                [Relevant Information] 
                All or part of same level nodes(no more than 10): ##1## 

                [Entry] 
                ##5##
                """
                
                
                prompt_leaf = """
                turn the [Entry] to text based on [Requirements] and [Relevant Information]. 
                
                [Requirements]
                The rewritten text should include all the relevant information of its surrounding nodes for easy retrieval.

                [Relevant Information] 
                All or part of same level nodes(no more than 10): ##1## 
                Tip: The meaning of ##3##: ##4## 

                [Entry] 
                ##5##
                """
                siblings = []
                if node.parent is not None:
                    if not node.is_leaf:
                        siblings = [
                            child.parent.name + ">>" + child.name
                            for child in node.parent.children
                            if child != node
                        ]
                    else:
                        siblings = [
                            p_sibling.name + ">>" + p_sibling.children[0].name
                            for p_sibling in node.parent.parent.children
                            if p_sibling != node.parent and len(p_sibling.children) == 1
                        ]

                u = node
                path = [u.name]
                while u.parent is not None:
                    u = u.parent
                    path.append(u.name)
                path.reverse()

                paths.append(">>".join(path))

                children = [child.name for child in node.children]

                if len(siblings) > 10:
                    siblings = random.sample(siblings, 10)
                if len(children) > 10:
                    children = random.sample(children, 10)

                sib_tokens = 0

                # print("Node:", node.name)
                # print("Path:", ">>".join(path))
                # print("Siblings:", str(siblings))

                if len(path) == 1:
                    prompt = prompt_root
                if node.is_leaf:
                    prompt = prompt_leaf
                if "##1##" in prompt:
                    prompt = prompt.replace("##1##", str(siblings))
                # if "##2##" in prompt:
                #     prompt = prompt.replace("##2##", str(children))
                if "Tip" in prompt:
                    prompt = prompt.replace("##3##", ">>".join(path[:-1]))
                    prompt = prompt.replace("##4##", x2desc[">>".join(path[:-1])])
                prompt = prompt.replace("##5##", ">>".join(path))

                item = {"prompt": prompt, "query": ">>".join(path)}
                if "Sub-level" not in prompt:
                    with open("leaves.json", "a") as f:
                        f.write(json.dumps(item) + "\n")
                if node.is_leaf:
                    all_leaf.append(item)

                prompts.append(item)

                if not leaf:
                    if prompts[0]["query"] not in x2desc.keys():
                        if len(prompts) >= 1000:
                            results = run(prompts)
                            for res in results:
                                x2desc[res["query"]] = res["output"]
                            prompts = []
                            with open(file_path, "w") as f:
                                json.dump(x2desc, f, indent=4, ensure_ascii=False)

                for child in node.children:
                    if leaf:
                        queue.append(child)
                    else:
                        if not child.is_leaf:
                            queue.append(child)
            # for prompt in prompts:
            #     if prompt["query"] not in x2desc.keys():
            #         tmp.append(prompt)
            # if len(tmp):
            #     print(len(tmp))
            #     results = run(tmp)
            #     for res in results:
            #         x2desc[res["query"]] = res["output"]
            #     tmp = []

            if not leaf:
                if len(prompts) and prompts[0]["query"] not in x2desc.keys():
                    results = run(prompts)
                    for res in results:
                        x2desc[res["query"]] = res["output"]

            if not leaf:
                while True:
                    with open(file_path, "w") as f:
                        json.dump(x2desc, f, indent=4, ensure_ascii=False)
                    with open(file_path, "r") as f:
                        path2nl_tmp = json.load(f)
                    if any(path not in path2nl_tmp.keys() for path in paths):
                        time.sleep(3)
                    else:
                        break
        if leaf:
            print("all_leaf: ", len(all_leaf))
            time.sleep(3)
            results = run(all_leaf)
            for result in results:
                x2desc[result["query"]] = result["output"]
            with open(file_path + "_results", "w") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
