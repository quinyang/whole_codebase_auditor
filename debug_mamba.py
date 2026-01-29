import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# --- FIX: SKIP SENSITIVE MAMBA LAYERS ---
# We compress the big layers (in_proj, out_proj) to 4-bit to save RAM.
# We keep the small, sensitive state layers (dt_proj, x_proj) in 16-bit to prevent the crash.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    llm_int8_skip_modules=["dt_proj", "x_proj"]  # <--- THE MAGIC FIX
)

MODEL_ID = "state-spaces/mamba-2.8b-hf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"⚙️ Loading Mamba (4-bit optimized) on {DEVICE}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    quantization_config=bnb_config, 
    device_map="auto"
)

# 2. Dummy File (Small Context)
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
attention_mask = inputs.attention_mask.to(DEVICE)

with torch.inference_mode():
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=attention_mask, # Fix warning
        max_new_tokens=100,
        do_sample=True,
        temperature=0.1, 
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )

print("\n" + "="*30)
print("OUTPUT")
print("="*30)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)

# Verify if it worked
if "admin_password_123" in result or "password" in result:
    print("\n✅ SUCCESS: 4-bit Mamba is working!")
else:
    print("\n⚠️  WARNING: Output looks generic. Model might need Instruction Tuning.")