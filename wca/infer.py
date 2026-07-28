"""Stage 5 -- Inference.

Model choice: `tiiuae/Falcon3-Mamba-7B-Instruct`. Pure Mamba-1 (64 decoder
blocks, no attention anywhere), so the "O(L) whole-repo pass" claim survives,
but *instruction-tuned* with a chat template. The previous `mamba-2.8b-hf` is a
base LM -- the old prompt ended in "1. [CRITICAL] Hardcoded Secret:" and the
model simply continued that text. It would have produced a confident finding for
an empty file. That is text continuation, not auditing.

Two hardware details that bite on Colab:

  * T4 is Turing (sm_75) and has no real bf16. `bnb_4bit_compute_dtype=bfloat16`
    silently emulates and runs slow. Pick dtype from compute capability.
  * `mamba-ssm` / `causal_conv1d` are OPTIONAL. Without them transformers uses
    the slower eager path, which is correct. Do not `pip install mamba-ssm` --
    it is a source build and fails on Colab routinely. Get correctness on eager
    first; only then install a prebuilt wheel pinned to the exact
    cuda/torch/abi/python combo.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

DEFAULT_MODEL = "tiiuae/Falcon3-Mamba-7B-Instruct"
# Falcon3-Mamba-7B-Instruct was trained at 32k. Going beyond is extrapolation.
MODEL_MAX_CONTEXT = 32_768

SYSTEM_PROMPT = """You are a security code auditor. You are given the source of one repository, \
concatenated into a single stream. Files marked mode="signature" show only imports and \
declarations; their bodies are elided.

Your job is to find CROSS-FILE vulnerabilities: defects that require reading two or more \
files to see. Examples: a credential defined in one file and transmitted or logged in \
another; unvalidated input entering at one boundary and reaching a dangerous sink in a \
different module; an authorization check in one layer that a second code path bypasses.

Rules:
- Report ONLY issues you can point at with a specific file path and a verbatim code line \
copied from the stream.
- A finding that involves only one file is out of scope unless it is a hardcoded secret.
- If you find nothing, return an empty array. Do not invent findings.
- Output valid JSON and nothing else."""

USER_TEMPLATE = """{context}

Audit the repository above.

Return a JSON array. Each element:
{{
  "title": "short description",
  "severity": "critical" | "high" | "medium" | "low",
  "category": "hardcoded_secret" | "injection" | "auth_bypass" | "unsafe_deserialization" | \
"path_traversal" | "data_exposure" | "other",
  "files": ["path/one.py", "path/two.py"],
  "evidence": "one verbatim line copied exactly from the stream",
  "why_cross_file": "how the two files combine to create the issue",
  "confidence": 0.0 to 1.0
}}

JSON array:"""


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    output_tokens: int
    prefill_seconds: float
    total_seconds: float

    def stats_line(self) -> str:
        tps = self.output_tokens / max(self.total_seconds - self.prefill_seconds, 1e-6)
        return (
            f"prefill {self.prompt_tokens:,} tok in {self.prefill_seconds:.1f}s; "
            f"decoded {self.output_tokens} tok at {tps:.1f} tok/s"
        )


def select_dtype():
    """fp16 on Turing (T4), bf16 on Ampere+ (L4/A100). CPU -> fp32."""
    import torch

    if not torch.cuda.is_available():
        return torch.float32, "cpu"
    major, _minor = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name(0)
    if major >= 8:
        return torch.bfloat16, name
    return torch.float16, name


def describe_environment() -> str:
    try:
        import torch
    except ImportError:
        return "torch not installed (install the 'gpu' extra)"
    dtype, dev = select_dtype()
    lines = [f"device: {dev}", f"compute dtype: {dtype}"]
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        lines.append(f"vram: {total:.1f} GB")
    for mod in ("mamba_ssm", "causal_conv1d"):
        try:
            __import__(mod)
            lines.append(f"{mod}: present (fast kernels)")
        except ImportError:
            lines.append(f"{mod}: absent (eager path -- correct, slower)")
    return "\n".join(lines)


class MambaAuditor:
    """Loads an instruction-tuned SSM once and runs audits against it."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        *,
        load_in_4bit: bool = True,
        device_map: str = "auto",
        hf_token: str | None = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.dtype, device_name = select_dtype()
        token = hf_token or os.getenv("HF_TOKEN")

        print(f"[wca] loading {model_id} on {device_name} (compute dtype {self.dtype})")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs: dict = {"device_map": device_map, "token": token}
        if load_in_4bit and torch.cuda.is_available():
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=self.dtype,  # NOT hardcoded bfloat16
            )
        else:
            kwargs["torch_dtype"] = self.dtype

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.model.eval()

    # ---- prompting ---------------------------------------------------------

    def build_prompt(self, context: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(context=context)},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    # ---- generation --------------------------------------------------------

    def generate(self, context: str, *, max_new_tokens: int = 1024) -> GenerationResult:
        import torch

        prompt = self.build_prompt(context)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        n_prompt = int(inputs.input_ids.shape[-1])

        if n_prompt > MODEL_MAX_CONTEXT:
            print(
                f"[wca] WARNING: {n_prompt:,} tokens exceeds the model's trained "
                f"{MODEL_MAX_CONTEXT:,} context. Lower --budget."
            )

        t0 = time.perf_counter()
        with torch.inference_mode():
            # Greedy. Auditing is a factual task -- sampling here manufactures
            # findings, which is precisely the failure mode of the old script.
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        total = time.perf_counter() - t0

        new_ids = out[0][n_prompt:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        n_out = int(new_ids.shape[-1])
        # Rough split: decode is ~linear in output length, so back it out.
        prefill = total * (1.0 - n_out / max(n_out + 8, 1)) if n_out else total

        return GenerationResult(
            text=text,
            prompt_tokens=n_prompt,
            output_tokens=n_out,
            prefill_seconds=prefill,
            total_seconds=total,
        )
