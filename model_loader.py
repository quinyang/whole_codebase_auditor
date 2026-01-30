"""
Model Loader Module
===================
Handles loading and configuring different Mamba-based models.

Key Features:
1. Automatic quantization configuration per model type
2. Proper handling of Mamba-specific layers during quantization
3. Memory-efficient loading strategies
4. Fallback mechanisms for different hardware configurations
"""

import os
import gc
import warnings
from typing import Optional, Tuple, Any, Dict
from dataclasses import dataclass

import torch

from config import WCAConfig, ModelBackend, ModelConfig


@dataclass
class LoadedModel:
    """Container for a loaded model and its tokenizer."""
    model: Any
    tokenizer: Any
    config: ModelConfig
    device: str
    actual_dtype: torch.dtype
    is_quantized: bool
    max_context: int


def get_device() -> str:
    """Determine the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def clear_memory():
    """Clear GPU and CPU memory caches."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_quantization_config(
    config: WCAConfig,
    model_config: ModelConfig,
) -> Optional[Any]:
    """
    Create the appropriate quantization configuration.
    
    CRITICAL: Different models have different sensitive layers that
    should NOT be quantized to avoid numerical instability.
    """
    if config.quantization_bits is None or not model_config.supports_quantization:
        return None
    
    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        warnings.warn(
            "bitsandbytes not available. Running in full precision. "
            "Install with: pip install bitsandbytes"
        )
        return None
    
    # Determine compute dtype
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    if config.quantization_bits == 4:
        # 4-bit quantization config
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=config.use_double_quant,
            bnb_4bit_quant_type="nf4",  # Normalized Float 4 - better for weights
            # CRITICAL: Skip sensitive Mamba layers
            # These layers have small tensors that lose too much precision when quantized
            llm_int8_skip_modules=model_config.quantization_skip_modules,
        )
    elif config.quantization_bits == 8:
        # 8-bit quantization config
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,  # Outlier threshold
            llm_int8_skip_modules=model_config.quantization_skip_modules,
        )
    else:
        warnings.warn(f"Unsupported quantization bits: {config.quantization_bits}. Using full precision.")
        return None


def load_model(
    config: Optional[WCAConfig] = None,
    verbose: bool = True,
) -> LoadedModel:
    """
    Load a model based on the configuration.
    
    Handles:
    - Model selection (Mamba, Zamba2, Codestral, Falcon)
    - Quantization setup
    - Device placement
    - Tokenizer configuration
    
    Returns:
        LoadedModel with model, tokenizer, and metadata
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    config = config or WCAConfig()
    model_config = config.model_config
    device = get_device()
    
    if verbose:
        print(f"⚙️  Loading {model_config.model_id}")
        print(f"   Device: {device}")
        print(f"   Quantization: {config.quantization_bits}-bit" if config.quantization_bits else "   Quantization: Full precision")
    
    # Clear memory before loading
    clear_memory()
    
    # Get quantization config
    quant_config = get_quantization_config(config, model_config)
    is_quantized = quant_config is not None
    
    # Determine dtype
    if device == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif device == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    # Load tokenizer first (lightweight)
    if verbose:
        print("   Loading tokenizer...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_id,
        trust_remote_code=True,  # Required for some models
    )
    
    # CRITICAL: Set pad token
    # Mamba models often don't have a pad token defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        if verbose:
            print("   ℹ️  Set pad_token = eos_token")
    
    # Load model
    if verbose:
        print("   Loading model weights...")
    
    load_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    
    if quant_config:
        load_kwargs["quantization_config"] = quant_config
        load_kwargs["device_map"] = "auto"  # Required for quantization
    else:
        load_kwargs["torch_dtype"] = dtype
        if device == "cuda":
            load_kwargs["device_map"] = "auto"
        # For CPU/MPS, we'll move manually
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_config.model_id,
            **load_kwargs,
        )
    except Exception as e:
        # Fallback: try without quantization
        if quant_config:
            warnings.warn(f"Quantization failed: {e}. Retrying without quantization.")
            load_kwargs.pop("quantization_config", None)
            load_kwargs["torch_dtype"] = dtype
            model = AutoModelForCausalLM.from_pretrained(
                model_config.model_id,
                **load_kwargs,
            )
            is_quantized = False
        else:
            raise
    
    # Move to device if needed (for non-quantized, non-auto-mapped models)
    if not hasattr(model, "hf_device_map") and device != "cpu":
        try:
            # Create explicit device object and move model
            target_device = torch.device(device)
            model = model.to(target_device)  # type: ignore[assignment]
        except Exception as e:
            warnings.warn(f"Failed to move model to {device}: {e}")
            device = "cpu"
    
    # Set to eval mode
    model.eval()
    
    # Configure model's pad token id
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    
    # Calculate actual max context
    # Some models report longer context than they were trained on
    actual_max_context = min(
        getattr(model.config, "max_position_embeddings", model_config.max_context_length),
        model_config.max_context_length,
    )
    
    if verbose:
        print(f"   ✅ Model loaded successfully!")
        print(f"   Max context: {actual_max_context:,} tokens")
        if is_quantized:
            print(f"   Memory usage: ~{torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        config=model_config,
        device=device,
        actual_dtype=dtype,
        is_quantized=is_quantized,
        max_context=actual_max_context,
    )


def estimate_memory_requirement(config: WCAConfig) -> Dict[str, float]:
    """
    Estimate memory requirements for a given configuration.
    
    Returns dict with 'model_gb', 'inference_gb', 'total_gb'.
    """
    model_config = config.model_config
    
    # Rough estimates based on parameter count
    # Assumes ~2 bytes per parameter for bf16
    param_estimates = {
        ModelBackend.MAMBA_PURE: 2.8e9,
        ModelBackend.ZAMBA2: 2.7e9,
        ModelBackend.CODESTRAL_MAMBA: 7e9,
        ModelBackend.FALCON_MAMBA: 7e9,
    }
    
    params = param_estimates.get(config.backend, 3e9)
    
    # Base model size
    if config.quantization_bits == 4:
        model_gb = params * 0.5 / 1e9  # ~0.5 bytes per param
    elif config.quantization_bits == 8:
        model_gb = params * 1.0 / 1e9  # ~1 byte per param
    else:
        model_gb = params * 2.0 / 1e9  # ~2 bytes per param
    
    # Inference overhead (KV cache, activations)
    # Mamba has O(1) state, much more efficient than transformers
    inference_gb = 2.0  # Roughly 2GB for Mamba inference
    
    return {
        "model_gb": model_gb,
        "inference_gb": inference_gb,
        "total_gb": model_gb + inference_gb,
    }


def check_hardware_compatibility(config: WCAConfig) -> Dict[str, Any]:
    """
    Check if the current hardware can run the specified configuration.
    
    Returns a dict with 'compatible', 'warnings', and 'recommendations'.
    """
    result = {
        "compatible": True,
        "warnings": [],
        "recommendations": [],
    }
    
    device = get_device()
    memory_req = estimate_memory_requirement(config)
    
    if device == "cpu":
        result["warnings"].append("Running on CPU will be very slow.")
        result["recommendations"].append("Consider using a GPU or Google Colab.")
    
    if device == "cuda":
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        if gpu_memory < memory_req["total_gb"]:
            result["compatible"] = False
            result["warnings"].append(
                f"Insufficient GPU memory: {gpu_memory:.1f}GB available, "
                f"~{memory_req['total_gb']:.1f}GB required."
            )
            result["recommendations"].append("Use 4-bit quantization (set quantization_bits=4)")
            
            if config.quantization_bits is None:
                result["recommendations"].append("Current config uses full precision.")
    
    return result