import torch
import torch.nn as nn
from model import GPT2
import tiktoken

'''
This file is supposed to generate text with the trained model.

As of Nov 30, 2024 (happy birthday, ChatGPT!):

I am still under the pretraining phase, so the model generates gibberish as the weights haven't been tuned yet.

More specifically, this is because the probabilities of the logits are all over the place and the model is not able to
generate coherent text. Cross-entropy loss needs to be calculated and backpropagated to update the weights, which I will
do from here.
'''

tokenizer = tiktoken.get_encoding('gpt2')

start_context = "Happy birthday to" # this is the input text, add any text you want here
encoded = tokenizer.encode(start_context)
print(f"\nencoded: {encoded}")

encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print(f"encoded_tensor.shape: {encoded_tensor.shape}\n")

GPT_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12, 
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

model = GPT2(GPT_CONFIG)

def generate_text(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        probabilities = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probabilities, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

# model eval
model.eval()

out = generate_text(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=40, # change this to determine the output text length
    context_size=GPT_CONFIG["context_length"]
)

print(f"Output: {out}")
print(f"Output length: {len(out[0])}\n")

decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(f"Decoded text:\n{decoded_text}")
