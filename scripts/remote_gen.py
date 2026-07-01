"""Remote GPU generation (fp16) for the precision-confound experiment (CRITIQUE_LOG iter-4 blocker #1).

Runs on a cloud GPU (e.g. AWS g5.xlarge, A10G 24GB) via vLLM. Generates ONLY raw completions for the
first N GSM8K test problems with Qwen2.5-7B-Instruct in **fp16**, matching the repo's rollout spec
exactly (same prompt, chat template, K, temperature, top_p, max_tokens). Grading/answer-extraction is
done LOCALLY afterwards with src/grader.py so the fp16 run is graded identically to the 4-bit run —
maximal comparability. Output: gsm8k_qwen-7b-fp16.raw.jsonl = {id, gold, question, generations:[K]}.

Usage on the instance:
  pip install -q vllm datasets
  python remote_gen.py --n 128 --k 16 --out gsm8k_qwen-7b-fp16.raw.jsonl
"""
import argparse
import json


PROMPT = ("Solve the problem. Reason step by step, then end with a single final line "
          "exactly in the form 'Answer: <answer>'.\n\nProblem: {q}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128, help="first N gsm8k test problems (match 7B subset)")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", default="gsm8k_qwen-7b-fp16.raw.jsonl")
    ap.add_argument("--max-tokens", type=int, default=1024)
    args = ap.parse_args()

    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    ds = load_dataset("openai/gsm8k", "main", split="test")
    probs = []
    for i in range(args.n):
        ex = ds[i]
        gold = ex["answer"].split("####")[-1].strip().replace(",", "")
        probs.append({"id": f"gsm8k-test-{i}", "gold": gold, "question": ex["question"]})

    llm = LLM(model=args.model, dtype="float16", gpu_memory_utilization=0.90,
              max_model_len=4096, seed=0)
    sp = SamplingParams(n=args.k, temperature=0.8, top_p=0.95, max_tokens=args.max_tokens)

    # vLLM applies the model's chat template via .chat()
    conversations = [[{"role": "user", "content": PROMPT.format(q=p["question"])}] for p in probs]
    outs = llm.chat(conversations, sp)

    with open(args.out, "w") as f:
        for p, o in zip(probs, outs):
            gens = [c.text for c in o.outputs]
            f.write(json.dumps({"id": p["id"], "gold": p["gold"],
                                "question": p["question"], "generations": gens}) + "\n")
    print(f"WROTE {args.out}: {len(probs)} problems x {args.k} samples (fp16)")


if __name__ == "__main__":
    main()
