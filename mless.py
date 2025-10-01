import gc
import json
import numpy as np
import torch
import torch.nn as nn
import random
import string
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from .math_equivalence import is_equiv
import argparse


def math_test_batch(args, model, tokenizer):
    path = '/research/projects/trans_llm/Zeru_Shi/dataset/Math-500/test.jsonl'
    with open(path, 'r') as f:
        dataset = [json.loads(line.strip()) for line in f]

    total_lines = len(dataset)
    ans_list = []
    start_time = time.time()


    for i in tqdm(range(0, total_lines, args.batch_size), desc="Batch inference"):
        batch = dataset[i:i+args.batch_size]

        prompts = [
            "Please reason step by step and put your final answer within \\boxed{{}}." +
           args.Mtokens +
            item["problem"]
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
                pad_token_id=tokenizer.eos_token_id
            )

        output_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        for out_text, gold in zip(output_texts, answers):
            isright = is_equiv(out_text, gold)
            ans_list.append(isright)

    true_ratio = sum(ans_list) / len(ans_list)
    elapsed = time.time() - start_time
    print(f"Accuracy: {true_ratio:.4f}, Time: {elapsed:.2f}s")
    return true_ratio


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Experiment Arguments")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Math-7B")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--Mtokens", type=str, default="/ / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / /")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True).to(args.device)
    model.eval()
    ratio = math_test_batch(args, model, tokenizer)