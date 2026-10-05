import torch
import torch.nn as nn
import random

# 1. Load and encode text
with open("input.txt", "r") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)

# 2. Define the Neural Network Architecture
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

hidden_size = 128
model = CharRNN(vocab_size, hidden_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()

# 3. Train the Model
print("Training your AI from scratch...")
seq_len = 25
epochs = 400

for epoch in range(epochs):
    start_idx = random.randint(0, len(data) - seq_len - 1)
    inputs = data[start_idx : start_idx + seq_len].unsqueeze(0)
    targets = data[start_idx + 1 : start_idx + seq_len + 1].unsqueeze(0)

    outputs, _ = model(inputs)
    loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.4f}")

# 4. Generate Text from the Trained AI
print("\n--- Generated AI Text ---")
model.eval()
start_char = "O"
input_tensor = torch.tensor([[char_to_idx.get(start_char, 0)]], dtype=torch.long)
hidden = None
generated_text = start_char

for _ in range(150):  # Generate 150 characters
    output, hidden = model(input_tensor, hidden)
    probs = torch.softmax(output[:, -1, :], dim=-1)
    next_char_idx = torch.multinomial(probs, num_samples=1).item()
    
    generated_text += idx_to_char[next_char_idx]
    input_tensor = torch.tensor([[next_char_idx]], dtype=torch.long)

print(generated_text)

# 5. Save the Model and Character Mappings
checkpoint = {
    "model_state": model.state_dict(),
    "chars": chars,
    "hidden_size": hidden_size
}
torch.save(checkpoint, "my_model.pth")
print("\nModel saved successfully as 'my_model.pth'!")