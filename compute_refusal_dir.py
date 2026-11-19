import random

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from tqdm import tqdm

torch.inference_mode()

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
# MODEL_ID = "Qwen/Qwen3-1.7B"
# MODEL_ID = "stabilityai/stablelm-2-zephyr-1_6b"
# MODEL_ID = "Qwen/Qwen1.5-1.8B-Chat"
# MODEL_ID = "Qwen/Qwen-1_8B-chat"
# MODEL_ID = "google/gemma-1.1-2b-it"
# MODEL_ID = "google/gemma-1.1-7b-it"
# MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID,
                                             trust_remote_code=True,
                                             dtype=torch.float16,
                                             device_map="cuda",
                                             quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                                                                                    bnb_4bit_compute_dtype=torch.float16))
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# settings:
instructions = 32
layer_idx = int(len(model.model.layers) * 0.6)
pos = -1

print("Instruction count: " + str(instructions))
print("Layer index: " + str(layer_idx))

with open("harmful.txt", "r") as f:
    harmful = f.readlines()

with open("harmless.txt", "r") as f:
    harmless = f.readlines()

harmful_instructions = random.sample(harmful, instructions)
harmless_instructions = random.sample(harmless, instructions)

harmful_toks = [
    tokenizer.apply_chat_template(conversation=[{"role": "user", "content": insn}],
                                  add_generation_prompt=True,
                                  return_tensors="pt") for insn in harmful_instructions]
harmless_toks = [
    tokenizer.apply_chat_template(conversation=[{"role": "user", "content": insn}],
                                  add_generation_prompt=True,
                                  return_tensors="pt") for insn in harmless_instructions]

max_its = instructions*2
bar = tqdm(total=max_its)


def generate(toks):
    # Extract input_ids from Hugging Face tokenizer output
    if hasattr(toks, "input_ids"):
        toks = toks.input_ids
    elif isinstance(toks, dict):
        toks = toks["input_ids"]

    # Ensure PyTorch tensor
    if not isinstance(toks, torch.Tensor):
        toks = torch.tensor(toks)

    toks = toks.to(model.device)
    # print("INPUT TYPE:", type(toks))
    # print("INPUT CONTENT:", toks)
    # print("INPUT IDS TYPE:", type(toks["input_ids"]) if hasattr(toks, "keys") and "input_ids" in toks else "No input_ids")

    return model.generate(
        input_ids=toks,
        use_cache=False,
        max_new_tokens=1,
        return_dict_in_generate=True,
        output_hidden_states=True
    )


# Generate hidden states
harmful_outputs = []
harmless_outputs = []

for toks in harmful_toks:
    harmful_outputs.append(generate(toks))
    bar.update(1)

for toks in harmless_toks:
    harmless_outputs.append(generate(toks))
    bar.update(1)

bar.close()

print("\nGeneration complete!")

# Extract hidden states
harmful_hidden = [
    output.hidden_states[0][layer_idx][:, pos, :]
    for output in harmful_outputs
]

harmless_hidden = [
    output.hidden_states[0][layer_idx][:, pos, :]
    for output in harmless_outputs
]

print("Harmful hidden states:", len(harmful_hidden))
print("Harmless hidden states:", len(harmless_hidden))
print("Hidden state shape:", harmful_hidden[0].shape)

# Calculate means
harmful_mean = torch.stack(harmful_hidden).mean(dim=0)
harmless_mean = torch.stack(harmless_hidden).mean(dim=0)

print("Harmful mean shape:", harmful_mean.shape)
print("Harmless mean shape:", harmless_mean.shape)

# Calculate refusal direction
refusal_dir = harmful_mean - harmless_mean
refusal_dir = refusal_dir / refusal_dir.norm()

print("Refusal direction shape:", refusal_dir.shape)

# Save
output_path = MODEL_ID.replace("/", "_") + "_refusal_dir.pt"

torch.save(refusal_dir.cpu(), output_path)

print(f"\nSuccessfully saved refusal direction to: {output_path}")
