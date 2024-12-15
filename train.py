import torch
import tiktoken
from dataloader import create_dataloader
from model import GPT2
from encoder_decoder import text_to_token_ids, token_ids_to_text
from cross_entropy_loss import calc_loss_batch, calc_loss_loader

# preprocessing
file_path = "the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

# training and eval data splitting
train_ratio = 0.9
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

# instantiating the train and val loaders
torch.manual_seed(3368332)

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

torch.manual_seed(123)
model = GPT2(GPT_CONFIG_124M)
model.eval()

train_loader = create_dataloader(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    shuffle=True,
    drop_last=True,
    num_workers=0
)

val_loader = create_dataloader(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    shuffle=True,
    drop_last=True,
    num_workers=0    
)

print("\n========================================= Checking loss at the beginning =========================================\n")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = tiktoken.get_encoding("gpt2")

model.to(device)

with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)

print("device:", device)
print("Training loss:", train_loss)
print("Validation loss:", val_loss)

# top-k sampling and temperature scaling
def generate(model, idx, max_new_tokens, context_size,
             temperature=0., top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        
        # getting the predictions
        with torch.no_grad():
            logits=model(idx_cond)

        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        # taking only the last logits since that's what we want!
        logits = logits[:, -1, :]

        # filtering logits with top_k sampling:
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                condition=logits < min_val,
                input=torch.tensor(float('-inf')).to(logits.device),
                other=logits
            )
        
        # applying temperature scaling
        if temperature > 0:
            logits = logits / temperature
            probabilities = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probabilities, num_samples=1)
        
        # applying greedy next-token generation
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        
        if idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)
    return idx


# ========================================= Model Train and Eval =========================================
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval() # disabling dropout for stable reproducible results
    with torch.no_grad(): # disables gradient tracking since this is eval
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate(
            model=model, idx=encoded, max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()

# ============= model training function ↓ ===============

def train_model_simple(model, train_loader, val_loader, optimizer, device,
                       num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # epoch loop
    for epoch in range(num_epochs):
        model.train()

        # batch loop
        for input_batch, target_batch in train_loader:
            # zero grad
            optimizer.zero_grad()
            
            # loss calculation on current batch
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )

            # backprop to calculate loss gradients
            loss.backward()

            # update the weights based on these loss gradients
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1

            # evaluation (prints the things that happen one-by-one during the training)
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"""\nEp {epoch+1} (Step {global_step:06d}): 
                      Train loss {train_loss:.3f}, 
                      Val loss {val_loss:.3f}""")
            
            # printing a sample text after each token
            generate_and_print_sample(
                model, tokenizer, device, start_context
            )
    return train_losses, val_losses, track_tokens_seen


print("\n========================================= Starting Training =========================================\n")

torch.manual_seed(123)

model = GPT2(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr = 0.0004, weight_decay=0.1)
num_epochs = 1

train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device, num_epochs=num_epochs,
    eval_freq=5, eval_iter=5, start_context="I wake up another day with the privilege", tokenizer=tokenizer
)

# ========================================= Saving and loading the model =========================================
torch.save(model.state_dict(), "model.pth")

model = GPT2(GPT_CONFIG_124M)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

torch.save(
    {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    },
    "model_and_optimizer.pth"
)
