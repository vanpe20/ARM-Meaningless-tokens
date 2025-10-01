import numpy as np
import torch
import torch.nn as nn
import random
import string

import matplotlib.pyplot as plt
from livelossplot import PlotLosses # pip install livelossplot
from tqdm import tqdm
import openai
from ..modeling_qwen2 import Qwen2ForCausalLM
from ..modeling_llama import LlamaForCausalLM
from ..modeling_gemma3 import Gemma3ForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from ..math_equivalence import is_equiv
import pandas as pd
import logging
import time
from .utils import choose_parameters, choose_model


path = './dataset/AIME2025/aime2025.jsonl''

os.makedirs("Mless_release/answer", exist_ok=True)

logging.basicConfig(
    filename="./answer/aime2025.log",        
    filemode="a",             
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO 
)
def math_generate(args, model, tokenizer):
    if 'gemma' in args.model_name:
        args.choose = False
    with open(path, "r") as f:
        dataset = [json.loads(line.strip()) for line in f]

    total_lines = len(dataset)
    ans_list = []

    for i in tqdm(range(0, total_lines, args.batch_size), desc="Batch inference"):
        batch = dataset[i:i+args.batch_size]

        prompts = [
            "Please reason step by step and put your final answer within \\boxed{{}}." + item["problem"]
            for item in batch
        ]
        answers = [item["answer"] for item in batch]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(args.device)
        input_ids = inputs["input_ids"]
        attn_masks = inputs["attention_mask"]

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=attn_masks,
                max_new_tokens=2048,
                range_c=args.range_c,
                percentage=args.percentage,
                pad_token_id=tokenizer.eos_token_id,
            )

        output_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        for out_text, gold in zip(output_texts, answers):
            isright = is_equiv(out_text, gold)
            ans_list.append(isright)

    true_ratio = sum(ans_list) / len(ans_list)
    logging.info(f"{args.model_name} with c={args.range_c}, p={args.percentage}, the score on Math500 is: {true_ratio:.4f}")

    return true_ratio

def comb(n, k):
    from math import comb as _comb
    return _comb(n, k)

def pass_at_k_from_counts(c: int, n: int, k: int) -> float:
    if k > n or n <= 0:
        return 0.0
    if c <= 0:
        return 0.0
    return 1.0 - comb(n - c, k) / comb(n, k)

def evaluate_pass_at3(
    args,
    tokenizer=None,
    model=None,
    max_new_tokens: int=512,
    samples_per_task: int=20, 
    # temperature: float=0.8,
    # top_p: float=0.95,
):
    k = 3
    assert samples_per_task >= k,

    per_task_scores = {}
    macro_sum = 0.0
    total_lines = sum(1 for _ in open(path, 'r'))

    with open(path, 'r') as file:
        for i, line in enumerate(tqdm(file, desc="Evaluating pass@3", total=total_lines)):
            item = json.loads(line.strip())
            prompt, ans = item['problem'], item['answer']

            system_prefix = "Please reason step by step and put your final answer within \\boxed{}."
            full_prompt = f"{system_prefix}{prompt}"

            inputs = tokenizer(full_prompt, return_tensors="pt").to(args.device)
            input_ids = inputs["input_ids"]
            attn_masks = inputs.get("attention_mask", None)
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attn_masks,
                    do_sample=True,
                    # temperature=temperature,
                    # top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=samples_per_task,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            for txt in texts:
                if txt.startswith(full_prompt):
                    txt = txt[len(full_prompt):]
                candidates.append(txt)

        c = 0
        for cand in candidates:
            try:
                ok = is_equiv(cand, ans)
            except Exception:
                ok = False
            c += int(ok)

        score = pass_at_k_from_counts(c, samples_per_task, k)
        task_id = i
        per_task_scores[task_id] = score
        macro_sum += score

    macro = macro_sum / len(per_task_scores) if per_task_scores else 0.0
    logging.info(f"Pass@3 on{args.model_name} with c={args.range_c}, p={args.percentage}, the score on AIME2025 is: pass@3_macro: {macro}, per_task: {per_task_score}")

def aime2025_pass3(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if args.range_c is None and args.percentage is None:
        args.range_c, args.percentage = choose_parameters(args.model_name)
        if args.range_c is not None and args.percentage is not None:
            model = choose_model(args.model_name).to(args.device)
        else:
            model = choose_model(args.model_name, arm = False).to(args.device)
    else:
        model = choose_model(args.model_name).to(args.device)
    model.eval()

    evaluate_pass_at3(args, tokenizer, model, samples_per_task = args.samples_per_task)

def aime2025_test(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if args.range_c is None and args.percentage is None:
        args.range_c, args.percentage = choose_parameters(args.model_name)
        if args.range_c is not None and args.percentage is not None:
            model = choose_model(args.model_name).to(args.device)
        else:
            model = choose_model(args.model_name, arm = False).to(args.device)
    else:
        model = choose_model(args.model_name).to(args.device)
    model.eval()

    math_generate(args, model, tokenizer)
            
