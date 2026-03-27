import tkinter as tk
from tkinter import ttk, scrolledtext
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math
import heapq

print("Loading Shakespeare Poem LLM GUI...")

SHAKESPEARE_POEMS = """Shall I compare thee to a summer's day?
Thou art more lovely and more temperate:
Rough winds do shake the darling buds of May,
And summer's lease hath all too short a date;
Sometime too hot the eye of heaven shines,
And often is his gold complexion dimm'd;
And every fair from fair sometime declines,
By chance or nature's changing course untrimm'd;
But thy eternal summer shall not fade,
Nor lose possession of that fair thou ow'st;
Nor shall death brag thou wander'st in his shade,
When in eternal lines to time thou grow'st:
So long as men can breathe or eyes can see,
So long lives this, and this gives life to thee."""


class CharDataset(Dataset):
    def __init__(self, text, seq_length):
        self.text = text
        self.seq_length = seq_length
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        
    def __len__(self):
        return len(self.text) - self.seq_length
    
    def __getitem__(self, idx):
        input_seq = self.text[idx:idx + self.seq_length]
        target_char = self.text[idx + self.seq_length]
        input_indices = [self.char_to_idx[c] for c in input_seq]
        target_idx = self.char_to_idx[target_char]
        return torch.tensor(input_indices), torch.tensor(target_idx)


class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super(CharRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size)
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        output = self.fc(output[:, -1, :])
        return output, hidden


class ShakespeareLLMGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Shakespeare Poem LLM - Thinking Path Visualizer")
        self.root.geometry("1100x700")
        
        self.model = None
        self.dataset = None
        self.tree_items = {}
        
        self.setup_ui()
        self.load_model()
    
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        title_label = ttk.Label(main_frame, text="Shakespeare Poem LLM - Thinking Path Visualizer", 
                                font=("Arial", 14, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=10)
        
        input_frame = ttk.LabelFrame(main_frame, text="Settings", padding="10")
        input_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(input_frame, text="Prompt:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.prompt_entry = ttk.Entry(input_frame, width=35)
        self.prompt_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        self.prompt_entry.insert(0, "Shall I compare thee")
        
        ttk.Label(input_frame, text="Depth Limit:").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.depth_var = tk.StringVar(value="10")
        depth_frame = ttk.Frame(input_frame)
        depth_frame.grid(row=0, column=3, sticky=tk.W, padx=5)
        ttk.Radiobutton(depth_frame, text="10", variable=self.depth_var, value="10").pack(side=tk.LEFT)
        ttk.Radiobutton(depth_frame, text="100", variable=self.depth_var, value="100").pack(side=tk.LEFT)
        ttk.Radiobutton(depth_frame, text="Unlimited", variable=self.depth_var, value="unlimited").pack(side=tk.LEFT)
        
        ttk.Label(input_frame, text="Temp:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.temp_scale = ttk.Scale(input_frame, from_=0.1, to=2.0, value=0.8, length=100)
        self.temp_scale.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(input_frame, text="Length:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        self.length_var = tk.StringVar(value="80")
        ttk.Entry(input_frame, textvariable=self.length_var, width=8).grid(row=1, column=3, sticky=tk.W, padx=5)
        
        input_frame.columnconfigure(1, weight=1)
        
        self.generate_btn = ttk.Button(main_frame, text="Generate Poem with Thinking Path", command=self.generate_poem)
        self.generate_btn.grid(row=2, column=0, columnspan=3, pady=10)
        
        self.progress_label = ttk.Label(main_frame, text="Ready - Model loaded", foreground="green", font=("Arial", 10))
        self.progress_label.grid(row=3, column=0, columnspan=3)
        
        paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=1)
        
        ttk.Label(left_frame, text="Thinking Path Tree", font=("Arial", 11, "bold")).pack(pady=5)
        
        tree_frame = ttk.Frame(left_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        self.thinking_tree = ttk.Treeview(tree_frame, height=20)
        self.thinking_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.thinking_tree.heading("#0", text="Thinking Steps")
        self.thinking_tree.column("#0", width=400)
        
        tree_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.thinking_tree.yview)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.thinking_tree.configure(yscrollcommand=tree_scroll.set)
        
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=1)
        
        ttk.Label(right_frame, text="Generated Poem", font=("Arial", 11, "bold")).pack(pady=5)
        self.poem_text = scrolledtext.ScrolledText(right_frame, height=20, wrap=tk.WORD, font=("Courier", 10))
        self.poem_text.pack(fill=tk.BOTH, expand=True)
        
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)
    
    def load_model(self):
        self.progress_label.config(text="Loading model...", foreground="blue")
        self.root.update()
        
        text = SHAKESPEARE_POEMS
        seq_length = 30
        embed_dim = 64
        hidden_dim = 128
        num_layers = 2
        
        self.dataset = CharDataset(text, seq_length)
        device = torch.device('cpu')
        
        self.model = CharRNN(self.dataset.vocab_size, embed_dim, hidden_dim, num_layers).to(device)
        
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005)
        
        dataloader = DataLoader(self.dataset, batch_size=16, shuffle=True, drop_last=True)
        
        epochs = 5
        self.progress_label.config(text=f"Training model ({epochs} epochs)...", foreground="orange")
        self.root.update()
        
        for epoch in range(epochs):
            total_loss = 0
            hidden = None
            for inputs, targets in dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                if hidden is not None:
                    hidden = hidden.detach()
                
                optimizer.zero_grad()
                outputs, hidden = self.model(inputs, hidden)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            self.progress_label.config(text=f"Training Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}", foreground="orange")
            self.root.update()
        
        self.model.eval()
        self.progress_label.config(text="Model ready!", foreground="green")
    
    def generate_poem(self):
        prompt = self.prompt_entry.get()
        if not prompt:
            return
        
        depth_str = self.depth_var.get()
        if depth_str == "unlimited":
            depth_limit = float('inf')
        else:
            depth_limit = int(depth_str)
        
        temperature = self.temp_scale.get()
        length = int(self.length_var.get())
        
        self.generate_btn.config(state=tk.DISABLED)
        self.thinking_tree.delete(*self.thinking_tree.get_children())
        self.poem_text.delete(1.0, tk.END)
        
        device = next(self.model.parameters()).device
        generated = prompt
        hidden = None
        
        for char_idx in range(length):
            input_indices = []
            for c in generated[-30:]:
                input_indices.append(self.dataset.char_to_idx.get(c, 0))
            
            while len(input_indices) < 30:
                input_indices.insert(0, 0)
            
            input_tensor = torch.tensor([input_indices[-30:]]).to(device)
            
            with torch.no_grad():
                output, hidden = self.model(input_tensor, hidden)
            
            logits = output[0] / temperature
            probs = torch.softmax(logits, dim=0)
            
            probs_list = probs.cpu().tolist()
            top_indices = heapq.nlargest(5, range(len(probs_list)), key=lambda i: probs_list[i])
            
            if char_idx < depth_limit:
                context = generated[-15:] if len(generated) > 15 else generated
                context_str = context.replace('\n', '\\n')
                
                parent_id = ''
                if char_idx > 0:
                    prev_items = self.thinking_tree.get_children()
                    if prev_items:
                        parent_id = prev_items[-1]
                
                step_text = f"[{char_idx+1}] Context: \"{context_str}\""
                item_id = self.thinking_tree.insert(parent_id, 'end', text=step_text, open=True)
                
                for rank, idx in enumerate(top_indices):
                    char = self.dataset.idx_to_char[idx]
                    prob = probs_list[idx]
                    
                    display_char = repr(char) if char == '\n' else char
                    child_text = f"  {rank+1}. '{display_char}' = {prob*100:.1f}%"
                    
                    if rank == 0:
                        child_text += " [SELECTED]"
                    
                    self.thinking_tree.insert(item_id, 'end', text=child_text)
            
            if char_idx % 10 == 0:
                self.progress_label.config(text=f"Generating step {char_idx+1}/{length}...", foreground="blue")
                self.root.update()
            
            char_idx_pred = torch.multinomial(probs, 1).item()
            predicted_char = self.dataset.idx_to_char[char_idx_pred]
            generated += predicted_char
        
        self.poem_text.delete(1.0, tk.END)
        self.poem_text.insert(1.0, generated)
        
        self.progress_label.config(text=f"Generation complete! {length} characters.", foreground="green")
        self.generate_btn.config(state=tk.NORMAL)


def main():
    root = tk.Tk()
    app = ShakespeareLLMGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
