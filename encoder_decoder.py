import torch

# utility functions for text to token ID and vice versa conversions
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # adding batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flatten = token_ids.squeeze(0) # removing batch dimension
    return tokenizer.decode(flatten.tolist())
