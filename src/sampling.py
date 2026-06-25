"""Samplers: FakeSampler (offline), HFSampler (transformers), VLLMSampler (GPU fast path).
See CLAUDE.md section 4 and section 5 (VLLMSampler untested locally)."""
import hashlib
from . import config, data


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
    """transformers backend. Works on CPU (slow) or GPU. Applies the chat template."""

    def __init__(self, model_key, **_):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_id = config.MODELS[model_key]
        self.tok = AutoTokenizer.from_pretrained(model_id, cache_dir=config.HF_CACHE)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, cache_dir=config.HF_CACHE,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self.s = config.SAMPLING

    def sample(self, problems, n):
        import torch
        out = []
        for p in problems:
            msgs = [{"role": "user", "content": data.build_prompt(p)}]
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = self.tok(text, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                gen = self.model.generate(
                    **inputs, do_sample=True, num_return_sequences=n,
                    temperature=self.s["temperature"], top_p=self.s["top_p"],
                    max_new_tokens=self.s["max_new_tokens"],
                    pad_token_id=self.tok.eos_token_id,
                )
            decoded = self.tok.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            out.append(decoded)
        return out


class VLLMSampler:
    """vLLM backend — GPU only, fast batched sampling. UNTESTED locally (see CLAUDE.md S5)."""

    def __init__(self, model_key, quantization=None, **_):
        from vllm import LLM
        from transformers import AutoTokenizer
        model_id = config.MODELS[model_key]
        self.tok = AutoTokenizer.from_pretrained(model_id, cache_dir=config.HF_CACHE)
        self.llm = LLM(model=model_id, quantization=quantization, download_dir=config.HF_CACHE)
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
