"""Samplers: FakeSampler (offline), HFSampler (transformers), VLLMSampler (GPU fast path).
See CLAUDE.md section 4 and section 5 (VLLMSampler untested locally)."""
import hashlib
import time
from . import config, data
from .logutil import log


def _seeded(*parts):
    """Deterministic pseudo-random in [0,1) from string parts (FakeSampler only)."""
    h = hashlib.md5("::".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


class FakeSampler:
    """No model. Emits 'Answer: <x>' where x is gold ~70% of the time, else a wrong value.
    Lets the whole pipeline (grade -> eval -> plots) run offline with zero downloads."""

    def __init__(self, correct_rate=0.7, **_):
        self.correct_rate = correct_rate

    def sample(self, problems, n):
        out = []
        for p in problems:
            samples = []
            for j in range(n):
                if _seeded(p.id, j) < self.correct_rate:
                    ans = p.gold
                else:
                    ans = str(int(_seeded("wrong", p.id, j) * 100))
                samples.append(f"Let me reason... step step.\nAnswer: {ans}")
            out.append(samples)
        return out


class HFSampler:
    """transformers backend. Works on CPU (slow) or GPU. Applies the chat template.

    `gen_batch` caps how many sequences are generated per forward pass — keep it small for
    large models (7B/8B) on a 16 GB GPU to avoid OOM; n is split into gen_batch-sized chunks."""

    def __init__(self, model_key, gen_batch=16, quantization=None, **_):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_id = config.MODELS[model_key]
        self.tok = AutoTokenizer.from_pretrained(model_id, cache_dir=config.HF_CACHE)
        kw = dict(cache_dir=config.HF_CACHE,
                  device_map="auto" if torch.cuda.is_available() else None)
        # 4-bit lets 7B/8B fit a 16GB P100 (fp16 weights alone are ~15GB → OOM). Needs bitsandbytes.
        if quantization in ("4bit", "nf4"):
            from transformers import BitsAndBytesConfig
            log(f"[hf] loading {model_id} in 4-bit (nf4) — fits big models on 16GB")
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        else:
            kw["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
        self.s = config.SAMPLING
        self.gen_batch = int(gen_batch)

    def sample(self, problems, n):
        import torch
        out = []
        for pi, p in enumerate(problems):
            t0 = time.time()
            msgs = [{"role": "user", "content": data.build_prompt(p)}]
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = self.tok(text, return_tensors="pt").to(self.model.device)
            decoded = []
            remaining = n
            while remaining > 0:
                b = min(self.gen_batch, remaining)
                with torch.no_grad():
                    gen = self.model.generate(
                        **inputs, do_sample=True, num_return_sequences=b,
                        temperature=self.s["temperature"], top_p=self.s["top_p"],
                        max_new_tokens=self.s["max_new_tokens"],
                        pad_token_id=self.tok.eos_token_id,
                    )
                decoded += self.tok.batch_decode(
                    gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                remaining -= b
            out.append(decoded)
            # Per-problem heartbeat: this is the real slow inner loop on the GPU.
            log(f"    [hf] problem {pi + 1}/{len(problems)} ({p.id}): "
                f"{len(decoded)} samples in {time.time() - t0:.1f}s")
        return out


class VLLMSampler:
    """vLLM backend — GPU only, fast batched sampling. UNTESTED locally (see CLAUDE.md S5)."""

    def __init__(self, model_key, quantization=None, tensor_parallel=None, **_):
        import torch
        from vllm import LLM
        from transformers import AutoTokenizer
        model_id = config.MODELS[model_key]
        # Default to ALL visible GPUs (e.g. 2 on Kaggle's T4×2) so 7B/8B shard across them.
        tp = int(tensor_parallel) if tensor_parallel else max(1, torch.cuda.device_count())
        self.tok = AutoTokenizer.from_pretrained(model_id, cache_dir=config.HF_CACHE)
        log(f"[vllm] loading {model_id} tensor_parallel_size={tp} quantization={quantization}")
        self.llm = LLM(model=model_id, quantization=quantization, download_dir=config.HF_CACHE,
                       tensor_parallel_size=tp)
        self.s = config.SAMPLING

    def sample(self, problems, n):
        from vllm import SamplingParams
        prompts = []
        for p in problems:
            msgs = [{"role": "user", "content": data.build_prompt(p)}]
            prompts.append(self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        params = SamplingParams(n=n, temperature=self.s["temperature"],
                                top_p=self.s["top_p"], max_tokens=self.s["max_new_tokens"])
        results = self.llm.generate(prompts, params)
        return [[o.text for o in r.outputs] for r in results]


def get_sampler(backend, model_key=None, **kw):
    backend = backend.lower()
    if backend == "fake":
        return FakeSampler(**kw)
    if backend == "hf":
        return HFSampler(model_key, **kw)
    if backend == "vllm":
        return VLLMSampler(model_key, **kw)
    raise ValueError(f"unknown backend: {backend}")
