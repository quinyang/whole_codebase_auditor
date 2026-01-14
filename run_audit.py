import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from parser.tree_parser import repo_scan_parser, build_mamba_prompt