import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

print("=" * 80)
print("SHAKESPEARE POEM LLM TRAINING APPLICATION")
print("=" * 80)
print()

DATA_FILE = "shakespeare_poems.txt"
MODEL_FILE = "shakespeare_model.pth"

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
So long lives this, and this gives life to thee.
When to the sessions of sweet silent thought
I summon up remembrance of things past,
I sigh the lack of many a thing I sought,
And with old woes new wail my dear time's waste:
Then can I drown an eye, unused to tears,
For precious friends hid in death's dateless night,
And weep afresh love's long since cancell'd woe,
And moan the expense of many a vanish'd sight:
Then can I grieve at grievances foregone,
And heavily from woe to woe tell o'er
The sad account of fore-bemoaned moan,
Which I new pay as if not paid before.
But if the while I think on thee, dear friend,
All losses are restored and sorrows end.
Let me not to the marriage of true minds
Admit impediments. Love is not love
Which alters when it alteration finds,
Or bends with the remover to remove:
O no! it is an ever-fixed mark
That looks on tempests and is never shaken;
It is the star to every wandering bark,
Whose worth's unknown, although his height be taken.
Love's not Time's fool, though rosy lips and cheeks
Within his bending sickle's compass come:
Love alters not with his brief hours and weeks,
But bears it out even to the edge of doom.
If this be error and upon me proved,
I never writ, nor no man ever loved.
Two roads diverged in a yellow wood,
And sorry I could not travel both
And be one traveler, long I stood
And looked down one as far as I could
To where it bent in the undergrowth;
Then took the other, as just as fair,
And having perhaps the better claim,
Because it was grassy and wanted wear;
Though as for that the passing there
Had worn them really about the same,
And both that morning equally lay
In leaves no step had trodden black.
Oh, I kept the first for another day!
Yet knowing how way leads on to way,
I doubted if I should ever come back.
I shall be telling this with a sigh
Somewhere ages and ages hence:
Two roads diverged in a wood, and I—
I took the one less traveled by,
And that has made all the difference.
I met a traveller from an antique land,
Who said—"Two vast and trunkless legs of stone
Stand in the desert. . . . Near them, on the sand,
Half sunk a shattered visage lies, whose frown,
And wrinkled lip, and sneer of cold command,
Tell that its sculptor well those passions read
Which yet survive, stamped on these lifeless things,
The hand that mocked them, and the heart that fed;
And on the pedestal, these words appear:
My name is Ozymandias, King of Kings;
Look on my Works, ye Mighty, and despair!
Nothing beside remains. Round the decay
Of that colossal Wreck, boundless and bare
The lone and level sands stretch far away."""


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
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super(CharRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size)
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        output = self.dropout(output)
        output = self.fc(output[:, -1, :])
        return output, hidden


def train_model(text, epochs=10):
    print("=" * 80)
    print("STEP 1: PREPARING TRAINING DATA")
    print("=" * 80)
    print()
    
    seq_length = 50
    batch_size = 32
    embed_dim = 128
    hidden_dim = 256
    num_layers = 2
    dropout = 0.2
    learning_rate = 0.001
    
    print(f"[LOG] Text length: {len(text)} characters")
    print(f"[LOG] Sequence length: {seq_length}")
    print(f"[LOG] Batch size: {batch_size}")
    print(f"[LOG] Epochs: {epochs}")
    print(f"[LOG] Embed dimension: {embed_dim}")
    print(f"[LOG] Hidden dimension: {hidden_dim}")
    print()
    
    dataset = CharDataset(text, seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    print(f"[LOG] Vocabulary size: {dataset.vocab_size}")
    print(f"[LOG] Number of training sequences: {len(dataset)}")
    print(f"[LOG] Number of batches per epoch: {len(dataloader)}")
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[LOG] Using device: {device}")
    print()
    
    model = CharRNN(dataset.vocab_size, embed_dim, hidden_dim, num_layers, dropout).to(device)
    
    print("=" * 80)
    print("STEP 2: MODEL ARCHITECTURE")
    print("=" * 80)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[LOG] Total parameters: {total_params:,}")
    print()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print("=" * 80)
    print("STEP 3: TRAINING")
    print("=" * 80)
    print()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        hidden = None
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            if hidden is not None:
                hidden = hidden.detach()
            
            optimizer.zero_grad()
            outputs, hidden = model(inputs, hidden)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 20 == 0:
                print(f"[EPOCH {epoch+1:2d}/{epochs}] Batch {batch_idx:3d}/{len(dataloader)} | Loss: {loss.item():.4f} | Perplexity: {math.exp(loss.item()):.2f}")
        
        avg_loss = total_loss / len(dataloader)
        perplexity = math.exp(avg_loss)
        
        print()
        print(f"--- EPOCH {epoch+1:2d}/{epochs} SUMMARY ---")
        print(f"[LOG] Average Loss: {avg_loss:.4f}")
        print(f"[LOG] Perplexity: {perplexity:.2f}")
        
        with torch.no_grad():
            model.eval()
            test_input = dataset.text[:seq_length]
            test_indices = [dataset.char_to_idx[c] for c in test_input]
            test_tensor = torch.tensor([test_indices]).to(device)
            output, _ = model(test_tensor)
            probs = torch.softmax(output, dim=1)
            top_prob, top_char_idx = probs[0].max(0)
            predicted_char = dataset.idx_to_char[top_char_idx.item()]
            print(f"[LOG] Sample prediction: '{predicted_char}' (prob: {top_prob.item():.4f})")
        
        print()
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'char_to_idx': dataset.char_to_idx,
        'idx_to_char': dataset.idx_to_char,
    }, MODEL_FILE)
    
    print(f"[LOG] Model saved to: {MODEL_FILE}")
    print()
    
    return model, dataset


def generate_poem(model, dataset, prompt, length=300, temperature=0.8):
    print("=" * 80)
    print("STEP 4: GENERATING NEW POEM")
    print("=" * 80)
    print()
    
    print(f"[LOG] Prompt: \"{prompt}\"")
    print(f"[LOG] Generation length: {length} characters")
    print(f"[LOG] Temperature: {temperature}")
    print()
    
    device = next(model.parameters()).device
    model.eval()
    
    generated = prompt
    hidden = None
    
    print("=" * 80)
    print("THINKING PROCESS - STEP BY STEP")
    print("=" * 80)
    print()
    
    for char_idx in range(length):
        input_indices = []
        for c in generated[-50:]:
            input_indices.append(dataset.char_to_idx.get(c, 0))
        
        while len(input_indices) < 50:
            input_indices.insert(0, 0)
        
        input_tensor = torch.tensor([input_indices[-50:]]).to(device)
        
        with torch.no_grad():
            output, hidden = model(input_tensor, hidden)
        
        logits = output[0] / temperature
        probs = torch.softmax(logits, dim=0)
        
        if char_idx % 30 == 0 or char_idx < 5:
            top_k = 5
            top_probs, top_indices = probs.topk(top_k)
            print(f"\n[STEP {char_idx+1}] Analyzing next character:")
            print(f"  Input context: \"{generated[-20:]}\"")
            print(f"  Top {top_k} character choices:")
            for i in range(top_k):
                char = dataset.idx_to_char[top_indices[i].item()]
                prob = top_probs[i].item()
                bar = "#" * int(prob * 40)
                print(f"    '{char}': {prob:.4f} {bar}")
        
        char_idx_pred = torch.multinomial(probs, 1).item()
        predicted_char = dataset.idx_to_char[char_idx_pred]
        
        if char_idx % 30 == 0 or char_idx < 5:
            print(f"  --> Selected: '{predicted_char}' (index: {char_idx_pred})")
            print()
        
        generated += predicted_char
        
        if char_idx % 50 == 0 and char_idx > 0:
            print(f"[LOG] Progress: {char_idx}/{length} characters generated...")
    
    print()
    print("=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print()
    
    return generated


def main():
    print()
    print("+" + "=" * 78 + "+")
    print("|" + " " * 22 + "SHAKESPEARE POEM LLM" + " " * 27 + "|")
    print("|" + " " * 15 + "Character-level RNN Training & Generation" + " " * 9 + "|")
    print("+" + "=" * 78 + "+")
    print()
    
    text = SHAKESPEARE_POEMS
    
    print(f"[LOG] Loaded Shakespeare poems: {len(text)} characters")
    print()
    
    model, dataset = train_model(text, epochs=10)
    
    print("\n" + "=" * 80)
    print("NEW POEM GENERATION")
    print("=" * 80)
    print()
    
    prompts = [
        "Shall I compare thee",
        "Love is not love",
        "Two roads diverged",
    ]
    
    for i, prompt in enumerate(prompts):
        print(f"\n{'='*80}")
        print(f"GENERATING POEM {i+1}: \"{prompt}\"")
        print(f"{'='*80}\n")
        
        poem = generate_poem(model, dataset, prompt, length=250, temperature=0.8)
        
        print("\n" + "=" * 80)
        print(f"FINAL GENERATED POEM {i+1}")
        print("=" * 80)
        print(poem)
        print()
    
    print("\n" + "=" * 80)
    print("APPLICATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
