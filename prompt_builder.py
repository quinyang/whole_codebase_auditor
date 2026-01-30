"""
Prompt Builder Module
=====================
Constructs prompts optimized for security auditing with Mamba models.
"""

from typing import List, Optional, Iterator, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import re

from config import WCAConfig, ModelBackend
from repo_scanner import ScannedFile, ScanResult


@dataclass
class PromptChunk:
    """A chunk of the codebase with its prompt."""
    prompt: str
    files_included: List[str]
    token_estimate: int
    chunk_index: int
    total_chunks: int
    is_final: bool


# Security-relevant patterns for file prioritization
SECURITY_PATTERNS = [
    r"auth", r"login", r"password", r"secret", r"token", r"key",
    r"crypt", r"session", r"sql", r"query", r"exec", r"shell",
    r"command", r"api", r"config", r"env", r"credential",
]


def calculate_security_priority(file: ScannedFile) -> int:
    """Calculate priority score based on security relevance."""
    score = 0
    path_lower = file.path.lower()
    content_lower = file.content.lower() if file.content else ""
    
    for pattern in SECURITY_PATTERNS:
        if re.search(pattern, path_lower):
            score += 10
        matches = len(re.findall(pattern, content_lower))
        score += min(matches, 5)
    
    if file.path.endswith((".py", ".js", ".ts", ".go", ".java")):
        score += 5
    if file.has_imports:
        score += 3
    score += min(len(file.function_names), 10)
    
    return score


def format_file_block(file: ScannedFile, include_metadata: bool = True) -> str:
    """Format a single file for inclusion in the prompt."""
    header = f'<file path="{file.path}"'
    if include_metadata and file.language:
        header += f' language="{file.language}"'
    if include_metadata and file.has_syntax_errors:
        header += ' has_errors="true"'
    header += '>'
    
    content = file.content.strip() if file.content else "[EMPTY FILE]"
    return f"{header}\n{content}\n</file>"


def build_code_block(files: List[ScannedFile]) -> str:
    """Build the code block section of the prompt."""
    return '\n\n'.join(format_file_block(f) for f in files)


class PromptBuilder:
    """
    Builds prompts for security auditing with intelligent chunking.
    """
    
    def __init__(self, config: Optional[WCAConfig] = None):
        self.config = config or WCAConfig()
    
    def _get_template(self) -> Dict[str, str]:
        """Get the prompt template for the configured backend."""
        backend = self.config.backend
        
        if backend == ModelBackend.MAMBA_PURE:
            return {
                "style": "completion",
                "header": "The following is source code from a software repository:\n\n",
                "footer": "\n\n" + "="*40 + "\nSECURITY AUDIT REPORT\n" + "="*40 + "\nFiles Scanned: {file_count}\nVulnerabilities Found:\n\n1. [CRITICAL] Hardcoded Secret:",
            }
        
        elif backend == ModelBackend.ZAMBA2:
            return {
                "style": "instruction",
                "header": "<|system|>\nYou are a senior security researcher. Analyze code for vulnerabilities.\n<|endoftext|>\n<|user|>\nPerform a security audit:\n\n",
                "footer": "\n<|endoftext|>\n<|assistant|>\n# Security Audit Report\n\n## Vulnerabilities Found\n\n",
            }
        
        elif backend == ModelBackend.CODESTRAL_MAMBA:
            return {
                "style": "instruction",
                "header": "[INST] You are a code security auditor. Analyze this codebase for vulnerabilities (hardcoded secrets, injection, path traversal, etc). Report severity, location, description, and fix for each.\n\n",
                "footer": "\n\nProvide a comprehensive security audit. [/INST]\n\n# Security Audit Report\n\n## Findings\n\n",
            }
        
        else:  # FALCON_MAMBA
            return {
                "style": "instruction",
                "header": "User: Analyze this codebase for security vulnerabilities:\n\n",
                "footer": "\n\nAssistant: # Security Audit Report\n\n",
            }
    
    def build_single_prompt(
        self, 
        scan_result: ScanResult,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Build a single prompt containing the entire codebase."""
        template = self._get_template()
        max_tokens = max_tokens or self.config.model_config.max_context_length
        
        # Sort files by security priority
        sorted_files = sorted(
            scan_result.files,
            key=calculate_security_priority,
            reverse=True
        )
        
        # Build code block
        code_block = build_code_block(sorted_files)
        
        # Assemble prompt
        prompt = template["header"] + code_block + template["footer"].format(
            file_count=len(sorted_files)
        )
        
        return prompt
    
    def build_chunked_prompts(
        self,
        scan_result: ScanResult,
        max_tokens_per_chunk: Optional[int] = None,
    ) -> Iterator[PromptChunk]:
        """
        Build multiple prompts for large codebases.
        
        Yields PromptChunk objects that can be processed sequentially.
        """
        template = self._get_template()
        max_tokens = max_tokens_per_chunk or (
            self.config.model_config.max_context_length - 
            self.config.max_new_tokens - 
            500  # Buffer for prompt overhead
        )
        
        # Sort by priority
        sorted_files = sorted(
            scan_result.files,
            key=calculate_security_priority,
            reverse=True
        )
        
        # Calculate overhead
        header_tokens = len(template["header"]) // 4
        footer_tokens = len(template["footer"]) // 4
        overhead = header_tokens + footer_tokens
        available_tokens = max_tokens - overhead
        
        # Group files into chunks
        chunks: List[List[ScannedFile]] = []
        current_chunk: List[ScannedFile] = []
        current_tokens = 0
        
        for file in sorted_files:
            file_tokens = file.token_estimate + 50  # Overhead for XML tags
            
            if current_tokens + file_tokens > available_tokens and current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_tokens = 0
            
            current_chunk.append(file)
            current_tokens += file_tokens
        
        if current_chunk:
            chunks.append(current_chunk)
        
        total_chunks = len(chunks)
        
        # Generate prompts
        for idx, chunk_files in enumerate(chunks):
            code_block = build_code_block(chunk_files)
            
            # Add chunk indicator for multi-chunk analysis
            chunk_header = template["header"]
            if total_chunks > 1:
                chunk_header = f"[Chunk {idx + 1}/{total_chunks}]\n\n" + chunk_header
            
            prompt = chunk_header + code_block + template["footer"].format(
                file_count=len(chunk_files)
            )
            
            yield PromptChunk(
                prompt=prompt,
                files_included=[f.path for f in chunk_files],
                token_estimate=sum(f.token_estimate for f in chunk_files) + overhead,
                chunk_index=idx,
                total_chunks=total_chunks,
                is_final=(idx == total_chunks - 1),
            )
    
    def estimate_chunks_needed(self, scan_result: ScanResult) -> int:
        """Estimate how many chunks will be needed for a scan result."""
        max_tokens = (
            self.config.model_config.max_context_length - 
            self.config.max_new_tokens - 
            1000
        )
        total_tokens = scan_result.total_tokens_estimate
        return max(1, (total_tokens + max_tokens - 1) // max_tokens)