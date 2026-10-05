import gradio as gr
import torch
import torch.nn as nn
import random
import os

# --- Model Definition ---
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

MODEL_PATH = "my_model.pth"

# --- Training Function ---
def train_model(training_text, epochs, hidden_size, learning_rate, progress=gr.Progress()):
    if not training_text or len(training_text.strip()) < 20:
        return "⚠️ Please provide at least 20 characters of training text."

    chars = sorted(list(set(training_text)))
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}

    data = torch.tensor([char_to_idx[c] for c in training_text], dtype=torch.long)
    hidden_size = int(hidden_size)
    epochs = int(epochs)

    model = CharRNN(vocab_size, hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    seq_len = min(25, len(training_text) - 2)
    logs = [f"Vocabulary size: {vocab_size} unique characters.", f"Total text length: {len(training_text)} chars.\nStarting training..."]

    for epoch in progress.tqdm(range(epochs), desc="Training AI"):
        start_idx = random.randint(0, len(data) - seq_len - 1)
        inputs = data[start_idx : start_idx + seq_len].unsqueeze(0)
        targets = data[start_idx + 1 : start_idx + seq_len + 1].unsqueeze(0)

        outputs, _ = model(inputs)
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == epochs - 1:
            logs.append(f"Epoch {epoch + 1}/{epochs} | Loss: {loss.item():.4f}")

    # Save checkpoint
    checkpoint = {
        "model_state": model.state_dict(),
        "chars": chars,
        "hidden_size": hidden_size
    }
    torch.save(checkpoint, MODEL_PATH)
    logs.append(f"\n Training complete! Model saved as '{MODEL_PATH}'.")
    return "\n".join(logs)

# --- Generation Function ---
def generate_text(prompt, length, temperature):
    if not os.path.exists(MODEL_PATH):
        return "⚠️ No trained model found. Please train a model in the 'Train AI' tab first!"

    checkpoint = torch.load(MODEL_PATH, weights_only=False)
    chars = checkpoint["chars"]
    hidden_size = checkpoint["hidden_size"]
    vocab_size = len(chars)

    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    model = CharRNN(vocab_size, hidden_size)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    if not prompt:
        prompt = chars[0]

    hidden = None
    result = prompt

    # Warm-up hidden state
    for ch in prompt:
        idx = char_to_idx.get(ch, 0)
        inp = torch.tensor([[idx]], dtype=torch.long)
        output, hidden = model(inp, hidden)

    # Autoregressive generation
    for _ in range(int(length)):
        logits = output[:, -1, :] / max(temperature, 1e-4)
        probs = torch.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1).item()
        
        result += idx_to_char.get(next_idx, "")
        inp = torch.tensor([[next_idx]], dtype=torch.long)
        output, hidden = model(inp, hidden)

    return result

# --- UI Layout ---
with gr.Blocks(title="My Custom Cloud AI Studio", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 My Cloud Text AI Studio\nTrain your own neural network and generate text right from this interface!")

    with gr.Tabs():
        with gr.Tab("🏋️ Train AI"):
            gr.Markdown("### Feed training data to your AI")
            data_input = gr.Textbox(
                label="Training Text Dataset", 
                lines=10, 
                placeholder="Paste stories, code, poems, or dialogue here..."
            )
            with gr.Row():
                epochs_slider = gr.Slider(50, 2000, value=500, step=50, label="Training Epochs")
                hidden_slider = gr.Slider(32, 512, value=128, step=32, label="Hidden Layer Size")
                lr_slider = gr.Slider(0.001, 0.02, value=0.005, step=0.001, label="Learning Rate")

            train_btn = gr.Button("🚀 Start Training", variant="primary")
            train_output = gr.Textbox(label="Training Progress & Loss Log", lines=8)
            train_btn.click(
                train_model, 
                inputs=[data_input, epochs_slider, hidden_slider, lr_slider], 
                outputs=train_output
            )

        with gr.Tab("✍️ Generate Text"):
            gr.Markdown("### Prompt your trained AI")
            with gr.Row():
                prompt_input = gr.Textbox(label="Starting Prompt", placeholder="Once upon a time...")
                length_slider = gr.Slider(20, 500, value=150, step=10, label="Characters to Generate")
                temp_slider = gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Creativity (Temperature)")

            gen_btn = gr.Button("✨ Generate", variant="primary")
            gen_output = gr.Textbox(label="Generated AI Output", lines=8)
            gen_btn.click(
                generate_text, 
                inputs=[prompt_input, length_slider, temp_slider], 
                outputs=gen_output
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)