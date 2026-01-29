import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# Import both functions from your parser
from tree_parser import repo_scan_parser, build_mamba_prompt

MODEL_ID = "state-spaces/mamba-2.8b-hf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class PureMambaAuditor:
    def __init__(self):
        print(f"⚙️  Initializing Pure Mamba WCA on {DEVICE}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, 
            device_map=DEVICE, 
            torch_dtype=torch.bfloat16
        ).eval()
        
        # --- FIX #1: PADDING CONFIGURATION ---
        # Mamba does not have a pad token by default. 
        # We MUST set it to EOS (End of String) so the model ignores empty space.
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.eos_token_id

    def audit_codebase(self, full_context_string):
        print(f"\n🧠 Tokenizing entire codebase...")

        # --- FIX #2: PROMPT ENGINEERING (FORCE COMPLETION) ---
        # Instead of "Query: Answer:", we force a "Document Completion" format.
        # We start the report FOR the model, so it just has to fill in the blanks.
        
        # 1. The input code
        prompt = full_context_string
        
        # 2. The Trigger Phrase (The "Trick")
        # We append a header that looks like a professional report.
        prompt += "\n\n" + "="*40 + "\n"
        prompt += "SECURITY AUDIT REPORT\n"
        prompt += "Target: Whole Repository\n"
        prompt += "Vulnerabilities Found:\n"
        prompt += "="*40 + "\n"
        prompt += "1. [CRITICAL] Hardcoded Secret:" # We start the first bullet point

        # Tokenize
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=False
        )
        
        input_ids = inputs.input_ids.to(DEVICE)
        
        # --- FIX #3: EXPLICIT ATTENTION MASK ---
        # We explicitly tell the model "These tokens are real data".
        attention_mask = inputs.attention_mask.to(DEVICE)
        
        seq_len = input_ids.shape[1]
        print(f"📊 Total Input Size: {seq_len:,} tokens")
        print("🕵️‍♂️ WCA is auditing... (generating report)")
        
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask, # Critical for valid output
                max_new_tokens=500,
                
                # --- FIX #4: GENERATION PARAMETERS ---
                do_sample=True,
                temperature=0.2,       # LOW temperature = Logical, less creative/random
                top_p=0.95,            # Focus on high-probability words
                repetition_penalty=1.1, # Stop it from saying "The code is The code is"
                
                pad_token_id=self.tokenizer.eos_token_id, # Redundant safety check
                use_cache=True 
            )
            
        # Decode output
        report = self.tokenizer.decode(outputs[0][seq_len:], skip_special_tokens=True)
        
        # Re-attach the trigger phrase so the output looks complete
        return "1. [CRITICAL] Hardcoded Secret:" + report