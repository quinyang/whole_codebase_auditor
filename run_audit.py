import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# 1. Use 4-bit Quantization (The Speed Fix)
# This shrinks the model so your 12GB VRAM can handle the long context without swapping.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

MODEL_ID = "state-spaces/mamba-2.8b-hf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"⚙️ Loading Mamba (4-bit optimized) on {DEVICE}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# Fix padding
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    quantization_config=bnb_config, # Apply 4-bit compression
    device_map="auto"
)

# 2. Create a Dummy File (Small Context)
# We test with ONE file to ensure the model isn't broken.
dummy_code = """
def connect_to_db():
    # TODO: Remove this before production
    password = "admin_password_123" 
    return connect("db_url", password)
"""

prompt = f"""<file name="db.py">\n{dummy_code}\n</file>

========================================
SECURITY AUDIT REPORT
========================================
The following is a list of security vulnerabilities found in the code above:
1. [CRITICAL] Hardcoded Secret:"""

print("\n🧠 Running Sanity Check (Generation)...")

inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

with torch.inference_mode():
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.1, # Very low temp for facts
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )

print("\n" + "="*30)
print("OUTPUT")
print("="*30)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
