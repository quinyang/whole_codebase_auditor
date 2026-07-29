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


def free_gpu() -> None:
    """Drop unreferenced models and empty the CUDA caching allocator.

    Notebook-specific hazard: `auditor = MambaAuditor()` rebinds the name but the
    old model stays alive until GC runs, and even then torch's caching allocator
    holds the freed blocks. Re-running a load cell therefore silently doubles
    VRAM use. This makes the reclaim explicit.
    """
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass


def fast_path_available() -> bool:
    """True if a fused selective-scan kernel is importable.

    Without one, transformers falls back to `slow_forward`, which changes the
    memory characteristics of the model completely -- see
    `slow_path_bytes_per_token`.
    """
    for mod in ("mamba_ssm", "causal_conv1d"):
        try:
            __import__(mod)
        except ImportError:
            return False
    return True


def slow_path_bytes_per_token(model) -> int:
    """Activation bytes per context token on the eager (non-kernel) path.

    This is the single most important number for running on a small GPU, and it
    is not obvious from the architecture. `FalconMambaMixer.slow_forward` builds

        discrete_A = exp(A[None,:,None,:] * dt[:,:,:,None])
        # shape [batch, intermediate_size, seq_len, ssm_state_size], fp32

    i.e. it materialises the discretised recurrence for **every timestep at
    once** instead of scanning. So activation memory is *linear in context
    length*, not O(1) in state as the SSM formulation promises. For
    Falcon3-Mamba-7B (intermediate 8192, state 16, fp32) that is

        8192 * 16 * 4 = 524,288 bytes = 0.5 MiB per token, per tensor

    and `slow_forward` holds roughly three such tensors live (discrete_A,
    discrete_B, deltaB_u), so ~1.5 MiB/token. A 24k context therefore wants
    ~36 GiB of activations and cannot fit on a 16 GB card at any quantisation --
    quantising the *weights* does nothing here, because this is activation
    memory.

    The fused `selective_scan` kernel is precisely what avoids materialising
    this: it runs the scan in SRAM and never allocates the seq_len dimension.
    That is what makes the O(1)-state claim real in practice.
    """
    cfg = model.config
    hidden = getattr(cfg, "hidden_size", 4096)
    intermediate = getattr(cfg, "intermediate_size", None) or int(
        hidden * getattr(cfg, "expand", 2)
    )
    state = getattr(cfg, "state_size", 16)
    n_live_tensors = 3
    return intermediate * state * 4 * n_live_tensors


def estimate_max_context(model, *, safety: float = 0.8) -> int:
    """Largest context that fits in free VRAM on the eager path."""
    import torch

    if not torch.cuda.is_available():
        return MODEL_MAX_CONTEXT
    free, _total = torch.cuda.mem_get_info()
    per_token = slow_path_bytes_per_token(model)
    return max(int(free * safety / per_token), 512)


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
            lines.append(f"{mod}: absent (eager path)")
    if not fast_path_available():
        lines.append(
            "\nNOTE: without fused kernels transformers uses `slow_forward`, which\n"
            "materialises a [batch, intermediate, seq_len, state] tensor -- activation\n"
            "memory is LINEAR in context length (~1.5 MiB/token for a 7B Mamba).\n"
            "Weight quantisation does not help; this is activation memory.\n"
            "Try `pip install kernels` first, and keep the budget small until it works."
        )
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

        # A previously-loaded model in the same process still owns its VRAM;
        # rebinding `auditor = MambaAuditor()` in a notebook does NOT free it.
        # Re-running a cell therefore loads a second copy and OOMs at generate
        # time with a confusingly small allocation request.
        free_gpu()

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.model.eval()
        self._report_footprint()

    def _report_footprint(self) -> None:
        """Print actual VRAM use, so a silently-unquantised load is obvious."""
        import torch

        params = sum(p.numel() * p.element_size() for p in self.model.parameters())
        line = f"[wca] weights {params / 2**30:.2f} GiB"
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 2**30
            free, total = (x / 2**30 for x in torch.cuda.mem_get_info())
            line += f" | allocated {alloc:.2f} GiB | free {free:.2f}/{total:.2f} GiB"
            if alloc > 8.0:
                line += (
                    "\n[wca] WARNING: that is far more than a 4-bit 7B (~6 GiB). Either "
                    "quantisation\n       did not apply (is bitsandbytes installed?) or an "
                    "older model is still\n       resident. Restart the runtime."
                )
        print(line)

    def free(self) -> None:
        """Release the model and empty the CUDA cache."""
        self.model = None
        free_gpu()

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

        # Fail fast rather than OOMing several minutes into a prefill. On the
        # eager path activation memory is linear in context length, so the real
        # ceiling is usually far below the model's trained 32k.
        if not fast_path_available() and torch.cuda.is_available():
            safe = estimate_max_context(self.model)
            if n_prompt > safe:
                per_tok = slow_path_bytes_per_token(self.model) / 2**20
                need = n_prompt * per_tok / 1024
                free = torch.cuda.mem_get_info()[0] / 2**30
                raise RuntimeError(
                    f"Context of {n_prompt:,} tokens will not fit.\n"
                    f"  No fused kernel -> transformers uses slow_forward, which "
                    f"materialises a [batch, intermediate, seq_len, state] tensor.\n"
                    f"  Activation cost ~{per_tok:.2f} MiB/token -> "
                    f"~{need:.1f} GiB needed, {free:.1f} GiB free.\n"
                    f"  Fixes, in order:\n"
                    f"    1. budget_tokens<={safe:,} (proves the pipeline now)\n"
                    f"    2. pip install kernels   (fused kernels, no source build)\n"
                    f"    3. an L4/A100 runtime instead of a T4\n"
                    f"  Note: 4-bit quantisation does NOT help -- this is "
                    f"activation memory, not weights."
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
