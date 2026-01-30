"""
WCA Auditor Module
==================
The main auditor class that orchestrates code scanning and analysis.

This is the primary interface for running security audits.
"""

import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

import torch

from config import WCAConfig, ModelBackend
from model_loader import LoadedModel, load_model, clear_memory
from repo_scanner import RepoScanner, ScanResult, scan_local_directory
from prompt_builder import PromptBuilder, PromptChunk


@dataclass
class AuditFinding:
    """A single security finding."""
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    title: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    description: str = ""
    recommendation: str = ""
    cwe_id: Optional[str] = None


@dataclass 
class AuditReport:
    """Complete audit report."""
    findings: List[AuditFinding] = field(default_factory=list)
    raw_output: str = ""
    files_analyzed: int = 0
    total_tokens_processed: int = 0
    chunks_processed: int = 0
    inference_time_seconds: float = 0.0
    
    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "CRITICAL")
    
    @property
    def high_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "HIGH")
    
    def summary(self) -> str:
        return f"""
Audit Summary
=============
Files Analyzed: {self.files_analyzed}
Total Findings: {len(self.findings)}
  - Critical: {self.critical_count}
  - High: {self.high_count}
  - Medium: {sum(1 for f in self.findings if f.severity == 'MEDIUM')}
  - Low: {sum(1 for f in self.findings if f.severity == 'LOW')}
Tokens Processed: {self.total_tokens_processed:,}
Inference Time: {self.inference_time_seconds:.2f}s
"""


class WCAuditor:
    """
    The Whole-Codebase Auditor.
    
    Example usage:
        # Initialize
        config = WCAConfig(backend=ModelBackend.CODESTRAL_MAMBA)
        auditor = WCAuditor(config)
        
        # Scan and audit
        report = auditor.audit_repo()
        print(report.summary())
        print(report.raw_output)
    """
    
    def __init__(
        self,
        config: Optional[WCAConfig] = None,
        verbose: bool = True,
    ):
        self.config = config or WCAConfig()
        self.verbose = verbose
        self._model: Optional[LoadedModel] = None
        self._prompt_builder = PromptBuilder(self.config)
    
    def _log(self, message: str):
        """Print message if verbose mode is on."""
        if self.verbose:
            print(message)
    
    def load_model(self) -> LoadedModel:
        """Load the model (lazy loading on first use)."""
        if self._model is None:
            self._model = load_model(self.config, verbose=self.verbose)
        return self._model
    
    def unload_model(self):
        """Unload the model and free memory."""
        if self._model is not None:
            del self._model.model
            del self._model.tokenizer
            self._model = None
            clear_memory()
            self._log("🧹 Model unloaded, memory cleared")
    
    def _generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """Run generation on a prompt."""
        model = self.load_model()
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        
        # Tokenize
        inputs = model.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=model.max_context - max_new_tokens,
        )
        
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)
        
        input_length = input_ids.shape[1]
        self._log(f"📊 Input: {input_length:,} tokens")
        
        # Generate
        with torch.inference_mode():
            outputs = model.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                repetition_penalty=self.config.repetition_penalty,
                pad_token_id=model.tokenizer.pad_token_id,
                eos_token_id=model.tokenizer.eos_token_id,
                use_cache=True,
            )
        
        # Decode only the generated part
        generated_ids = outputs[0][input_length:]
        result = model.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return result
    
    def audit_scan_result(self, scan_result: ScanResult) -> AuditReport:
        """
        Audit a previously scanned codebase.
        
        Handles chunking automatically if the codebase is too large.
        """
        start_time = time.time()
        
        # Check if we need chunking
        chunks_needed = self._prompt_builder.estimate_chunks_needed(scan_result)
        
        if chunks_needed == 1:
            # Single pass
            self._log("🔍 Single-pass audit (codebase fits in context)")
            prompt = self._prompt_builder.build_single_prompt(scan_result)
            
            raw_output = self._generate(prompt)
            total_tokens = scan_result.total_tokens_estimate
            chunks_processed = 1
            
        else:
            # Multi-chunk
            self._log(f"🔍 Multi-chunk audit ({chunks_needed} chunks)")
            outputs = []
            total_tokens = 0
            
            for chunk in self._prompt_builder.build_chunked_prompts(scan_result):
                self._log(f"  Processing chunk {chunk.chunk_index + 1}/{chunk.total_chunks}")
                
                chunk_output = self._generate(chunk.prompt)
                outputs.append(f"## Chunk {chunk.chunk_index + 1}: {', '.join(chunk.files_included[:3])}...\n\n{chunk_output}")
                total_tokens += chunk.token_estimate
            
            raw_output = "\n\n---\n\n".join(outputs)
            chunks_processed = chunks_needed
        
        inference_time = time.time() - start_time
        
        # Parse findings (basic parsing, can be enhanced)
        findings = self._parse_findings(raw_output)
        
        return AuditReport(
            findings=findings,
            raw_output=raw_output,
            files_analyzed=len(scan_result.files),
            total_tokens_processed=total_tokens,
            chunks_processed=chunks_processed,
            inference_time_seconds=inference_time,
        )
    
    def audit_repo(self) -> AuditReport:
        """
        Scan and audit the configured GitHub repository.
        
        This is the main entry point for most use cases.
        """
        self._log("🚀 Starting Whole-Codebase Audit")
        self._log(f"   Backend: {self.config.backend.value}")
        self._log(f"   Model: {self.config.model_config.model_id}")
        
        # Scan
        self._log("\n📁 Scanning repository...")
        scanner = RepoScanner(self.config)
        scan_result = scanner.scan(verbose=self.verbose)
        
        # Audit
        self._log("\n🔍 Running security analysis...")
        return self.audit_scan_result(scan_result)
    
    def audit_local(self, directory: str) -> AuditReport:
        """Audit a local directory."""
        self._log(f"🚀 Auditing local directory: {directory}")
        
        scan_result = scan_local_directory(
            directory, 
            self.config, 
            verbose=self.verbose
        )
        
        return self.audit_scan_result(scan_result)
    
    def _parse_findings(self, raw_output: str) -> List[AuditFinding]:
        """
        Parse findings from raw model output.
        
        This is a basic parser - can be enhanced with regex or
        structured output parsing.
        """
        findings = []
        
        # Look for severity markers
        import re
        
        # Pattern: [SEVERITY] or **SEVERITY** followed by text
        patterns = [
            r'\[CRITICAL\][:\s]*(.+?)(?=\[(?:CRITICAL|HIGH|MEDIUM|LOW)\]|\Z)',
            r'\[HIGH\][:\s]*(.+?)(?=\[(?:CRITICAL|HIGH|MEDIUM|LOW)\]|\Z)',
            r'\[MEDIUM\][:\s]*(.+?)(?=\[(?:CRITICAL|HIGH|MEDIUM|LOW)\]|\Z)',
            r'\[LOW\][:\s]*(.+?)(?=\[(?:CRITICAL|HIGH|MEDIUM|LOW)\]|\Z)',
        ]
        
        severities = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
        
        for pattern, severity in zip(patterns, severities):
            matches = re.findall(pattern, raw_output, re.DOTALL | re.IGNORECASE)
            for match in matches:
                # Extract title (first line) and description
                lines = match.strip().split('\n')
                title = lines[0].strip() if lines else "Unknown"
                description = '\n'.join(lines[1:]).strip() if len(lines) > 1 else ""
                
                # Try to extract file path
                file_match = re.search(r'(?:File|Location|Path)[:\s]*([^\s:]+\.[a-z]+)', match, re.I)
                file_path = file_match.group(1) if file_match else None
                
                # Try to extract line number
                line_match = re.search(r'(?:line|:)(\d+)', match, re.I)
                line_number = int(line_match.group(1)) if line_match else None
                
                findings.append(AuditFinding(
                    severity=severity,
                    title=title[:100],  # Truncate long titles
                    file_path=file_path,
                    line_number=line_number,
                    description=description[:500],  # Truncate
                ))
        
        return findings


def quick_audit(
    repo: Optional[str] = None,
    local_dir: Optional[str] = None,
    backend: ModelBackend = ModelBackend.CODESTRAL_MAMBA,
    quantization_bits: int = 4,
) -> AuditReport:
    """
    Quick function to run an audit with minimal setup.
    
    Usage:
        # GitHub repo
        report = quick_audit(repo="owner/repo")
        
        # Local directory  
        report = quick_audit(local_dir="/path/to/code")
    """
    import os
    
    config = WCAConfig(
        backend=backend,
        quantization_bits=quantization_bits,
    )
    
    if repo:
        os.environ["GITHUB_REPO"] = repo
    
    auditor = WCAuditor(config)
    
    if local_dir:
        return auditor.audit_local(local_dir)
    else:
        return auditor.audit_repo()