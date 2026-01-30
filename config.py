"""
WCA Configuration Module
========================
Centralized configuration for the Whole-Codebase Auditor.

Key Design Decisions:
1. Use Zamba2 (Mamba-Attention Hybrid) with MUCH better instruction following
2. Configurable chunking strategy for when context exceeds model limits
3. Proper quantization settings for each model type
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from enum import Enum

class ModelBackend(Enum):
    """Supported model backends for WCA."""
    MAMBA_PURE = "mamba_pure"           # state-spaces/mamba-2.8b-hf (base, needs fine-tuning)
    ZAMBA2 = "zamba2"                   # Zyphra/Zamba2-2.7B (hybrid, better OOB)
    CODESTRAL_MAMBA = "codestral"       # mistralai/codestral-mamba (code-specific!)
    FALCON_MAMBA = "falcon_mamba"       # tiiuae/falcon-mamba-7b (larger, better quality)


@dataclass
class ModelConfig:
    """Configuration for a specific model backend."""
    model_id: str 
    max_context_length: int
    supports_quantization: bool
    quantization_skip_modules: List[str] = field(default_factory=list)
    recommended_dtype: str = "bfloat16"
    is_instruction_tuned: bool = False
    is_code_specialized: bool = False


# Model Registry - Add new models here
MODEL_REGISTRY: Dict[ModelBackend, ModelConfig] = {
    ModelBackend.MAMBA_PURE: ModelConfig(
        model_id="state-spaces/mamba-2.8b-hf",
        max_context_length=2048,  # Training context was limited
        supports_quantization=True,
        quantization_skip_modules=["dt_proj", "x_proj", "A_log", "D"],
        is_instruction_tuned=False,
        is_code_specialized=False,
    ),
    ModelBackend.ZAMBA2: ModelConfig(
        model_id="Zyphra/Zamba2-2.7B-instruct",  # Use instruct version!
        max_context_length=4096,
        supports_quantization=True,
        quantization_skip_modules=["dt_proj", "x_proj"],
        is_instruction_tuned=True,
        is_code_specialized=False,
    ),
    ModelBackend.CODESTRAL_MAMBA: ModelConfig(
        model_id="mistralai/Mamba-Codestral-7B-v0.1",
        max_context_length=256000,  # 256K context window!
        supports_quantization=True,
        quantization_skip_modules=[],  # Codestral handles quantization well
        is_instruction_tuned=True,
        is_code_specialized=True,  # <-- This is what you need!
    ),
    ModelBackend.FALCON_MAMBA: ModelConfig(
        model_id="tiiuae/falcon-mamba-7b-instruct",
        max_context_length=8192,
        supports_quantization=True,
        quantization_skip_modules=["dt_proj", "x_proj"],
        is_instruction_tuned=True,
        is_code_specialized=False,
    ),
}

@dataclass  
class WCAConfig:
    """
    Master configuration for the Whole-Codebase Auditor.
    
    Usage:
        config = WCAConfig(backend=ModelBackend.CODESTRAL_MAMBA)
        config = WCAConfig.from_env()  # Load from environment
    """
    # Model Selection
    backend: ModelBackend = ModelBackend.CODESTRAL_MAMBA # Default to Codestral Mamba
    
    # Quantization (set to None for full precision)
    quantization_bits: Optional[int] = 4  # 4, 8, or None   # 4 bits is standard
    use_double_quant: bool = True
    
    # Generation Parameters
    max_new_tokens: int = 1024
    temperature: float = 0.1  # Low for deterministic security analysis
    top_p: float = 0.95
    repetition_penalty: float = 1.15
    
    # Context Management
    chunk_overlap_tokens: int = 512  # Overlap between chunks for continuity
    max_files_per_chunk: int = 50    # Soft limit for chunking strategy
    
    # GitHub Settings (from environment)
    github_token: Optional[str] = field(default_factory=lambda: os.getenv("GITHUB_API_KEY"))
    github_repo: Optional[str] = field(default_factory=lambda: os.getenv("GITHUB_REPO"))
    target_branch: str = field(default_factory=lambda: os.getenv("TARGET_BRANCH", "main"))
    
    # File Filtering
    skip_patterns: List[str] = field(default_factory=lambda: [
        "*.lock", "*.min.js", "*.min.css", 
        "package-lock.json", "yarn.lock", "Cargo.lock",
        "json.hpp", "node_modules/*", "vendor/*", ".git/*"
    ])
    max_file_size_kb: int = 200  # Skip files larger than this
    
    @property
    def model_config(self) -> ModelConfig:
        """Get the configuration for the selected backend."""
        return MODEL_REGISTRY[self.backend]
    
    @classmethod
    def from_env(cls) -> "WCAConfig":
        """Create config from environment variables."""
        backend_str = os.getenv("WCA_BACKEND", "codestral").lower()
        backend_map = {
            "mamba": ModelBackend.MAMBA_PURE,
            "mamba_pure": ModelBackend.MAMBA_PURE,
            "zamba": ModelBackend.ZAMBA2,
            "zamba2": ModelBackend.ZAMBA2,
            "codestral": ModelBackend.CODESTRAL_MAMBA,
            "falcon": ModelBackend.FALCON_MAMBA,
        }
        backend = backend_map.get(backend_str, ModelBackend.CODESTRAL_MAMBA)
        
        return cls(
            backend=backend,
            quantization_bits=int(os.getenv("WCA_QUANT_BITS", "4")) or None,
            max_new_tokens=int(os.getenv("WCA_MAX_TOKENS", "1024")),
            temperature=float(os.getenv("WCA_TEMPERATURE", "0.1")),
        )
    

# Vulnerability Categories for structured output
VULN_CATEGORIES = {
    "HARDCODED_SECRET": {
        "severity": "CRITICAL",
        "description": "API keys, passwords, or tokens hardcoded in source",
        "cwe": "CWE-798",
    },
    "SQL_INJECTION": {
        "severity": "CRITICAL",
        "description": "Unsanitized user input in SQL queries",
        "cwe": "CWE-89",
    },
    "PATH_TRAVERSAL": {
        "severity": "HIGH",
        "description": "User input used in file paths without validation",
        "cwe": "CWE-22",
    },
    "INSECURE_DESERIALIZATION": {
        "severity": "HIGH",
        "description": "Untrusted data deserialized without validation",
        "cwe": "CWE-502",
    },
    "COMMAND_INJECTION": {
        "severity": "CRITICAL",
        "description": "User input passed to shell commands",
        "cwe": "CWE-78",
    },
    "CROSS_FILE_DATA_FLOW": {
        "severity": "MEDIUM",
        "description": "Sensitive data flows between modules without sanitization",
        "cwe": "CWE-200",
    },
    "RACE_CONDITION": {
        "severity": "MEDIUM",
        "description": "Time-of-check to time-of-use vulnerabilities",
        "cwe": "CWE-367",
    },
}