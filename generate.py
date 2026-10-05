import torch
import torch.nn as nn

# 1. Load the Saved Checkpoint
checkpoint = torch.load("my_model.pth", weights_only=False)
chars = checkpoint["chars"]
hidden_size = checkpoint["hidden_size"]
vocab_size = len(chars)

char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# 2. Recreate Model Architecture
class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden

# 3. Load Model Weights
model = CharRNN(vocab_size, hidden_size)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# 4. Generate Text
def generate(start_prompt="Once", length=200, temperature=0.7):
    hidden = None
    current_text = start_prompt
    
    # Warm up the hidden state with prompt characters
    for ch in start_prompt:
        idx = char_to_idx.get(ch, 0)
        input_tensor = torch.tensor([[idx]], dtype=torch.long)
        output, hidden = model(input_tensor, hidden)

    # Generate new characters
    for _ in range(length):
        logits = output[:, -1, :] / max(temperature, 1e-5)
        probs = torch.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1).item()
        
        current_text += idx_to_char[next_idx]
        input_tensor = torch.tensor([[next_idx]], dtype=torch.long)
        output, hidden = model(input_tensor, hidden)
        
    return current_text

prompt = input("Enter a starting word/character: ")
print("\n--- Output ---")
print(generate(start_prompt=prompt if prompt else "O"))