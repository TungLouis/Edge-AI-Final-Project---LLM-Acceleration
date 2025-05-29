import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
from datasets import load_dataset
import random
import numpy as np

def evaluate_ppl(model, tokenizer, device="cuda:0"):
    test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    
    test_enc = tokenizer("\n\n".join(test_dataset["text"]), return_tensors="pt")
    model.seqlen = 2048
    test_enc = test_enc.input_ids.to(device)
    
    nsamples = test_enc.numel() // model.seqlen
    nlls = []  
    for i in tqdm(range(nsamples), desc="Evaluating..."):
        batch = test_enc[:, (i * model.seqlen):((i + 1) * model.seqlen)]
        
        with torch.no_grad():
            lm_logits = model(batch).logits

        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = test_enc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    
    return ppl.item()


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-t","--throughput", action="store_true", help="Run throughput evaluation")
    group.add_argument("-p", "--ppl", action="store_true", help="Run perplexity evaluation")
    return parser.parse_args()

def main():
    ############## Set Up ##############
    torch.manual_seed(0)
    random.seed(0)
    
    max_new_tokens = 256    # Number of new tokens to generate
    device = 'cuda'
    ### === TODO: Load your model (you may change this part) ===
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 指定使用第 x 張 GPU
    from vllm import LLM, SamplingParams

    args = get_args()
    model_name = "awq_wbit4_gs128"

    if args.throughput:
        params = SamplingParams(
            min_tokens=250,
            max_tokens=max_new_tokens,

        )
        model = LLM(
            model_name,
            # gpu_memory_utilization=0.7,
            max_model_len=2048,
            dtype=torch.float16,
        )
    elif args.ppl:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
        )
    #####################################
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if args.throughput:

        warmup_prompt = "Explain what AI is."
        inputs = tokenizer(warmup_prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        for i in tqdm(range(5), desc="Warm Up..."):
            generated = model.generate(warmup_prompt, params)


        prompt = "How to learn a new language?"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        tputs = []
        time_record = []
        for _ in tqdm(range(10), desc="Test Inference"):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            generated = model.generate(prompt, params)


            end.record()
            torch.cuda.synchronize()
            elapsed_ms = start.elapsed_time(end)

            tput = len(generated[0].outputs[0].token_ids) / (elapsed_ms / 1000)
            time_record.append(elapsed_ms / 1000)
            tputs.append(tput)

        response = generated[0].outputs[0].text
        sorted_tputs = np.sort(tputs)[2:-2]
        org_tput = np.mean(sorted_tputs)
        print(f'Prompt: {prompt}\nResponse: {response}\n')
        
        print(f'Time Record: {time_record}')
        print(f'Throughput Record: {tputs} toks/s\n')

        ### Your final throughput result ###
        print(f'Throughput: {org_tput} toks/s')

    elif args.ppl:
        ppl = evaluate_ppl(model, tokenizer, device)
        print(f"Perplexity (PPL): {ppl}")

if __name__ == "__main__":
    main()