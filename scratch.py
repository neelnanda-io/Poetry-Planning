# %%
from neel.imports import *
import torch
from transformer_lens import HookedTransformer
torch.set_grad_enabled(False)

# Ignore CUDA architecture mismatch
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Set up the prompt
prompt = "A rhyming couplet:\nHe saw a carrot and had to grab it\nHis hunger was"

# # Load Gemma-9B model
# model = HookedTransformer.from_pretrained_no_processing("google/gemma-2-2b", device="cuda")


# # Run inference
# tokens = model.to_tokens(prompt)
# with torch.inference_mode():
#     logits = model(tokens)  # Using __call__ instead of forward
# next_token_logits = logits[0, -1]  # Get logits for next token prediction
# top_k = 20

# # Get top k completions
# values, indices = torch.topk(next_token_logits, top_k)
# tokens_list = indices.tolist()
# probs = torch.softmax(values, dim=0).tolist()

# print("Top completions for 'His hunger was':")
# for token, prob in zip(tokens_list, probs):
#     completion = model.to_string([token])  # Pass as list to satisfy type checker
#     print(f"{completion}: {prob:.3f}")

# # Generate a full completion
# generated = model.generate(tokens, max_new_tokens=10, temperature=0.7)
# print("\nFull completion:")
# print(model.to_string(generated[0]))

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.text import Text

# Initialize rich console
console = Console()

# Load model and tokenizer
hf_model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", device_map="cpu")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")

# Set up the prompt
prompt = "A rhyming couplet:\nHe saw a carrot and had to grab it\nHis hunger was"

# Generate completion
with torch.inference_mode():
    inputs = tokenizer(prompt, return_tensors="pt")  # Keep on CPU
    outputs = hf_model.generate(
        **inputs,
        max_new_tokens=20,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.convert_tokens_to_ids("\n"),  # Stop at newline
    )

# Get the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Format the output
text = Text()
text.append(prompt)  # Original prompt in normal text
generated_part = generated_text[len(prompt):]  # Extract only the generated part
text.append(generated_part, style="bold")  # Generated text in bold

# Print with a title
console.print("\n=== Hugging Face Gemma-2B Completion ===")
console.print(text)

# %%
