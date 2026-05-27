"""
Timeless: Modular Grokking & Mechanistic Interpretability Pipeline
Standalone PyTorch Research Script (Compatible with Google Colab & local CUDA)
"""

import os
import sys
import random
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataclasses import dataclass
import matplotlib.pyplot as plt

# ==========================================
# 1. ARCHITECTURE DEFINITIONS
# ==========================================

class HookPoint(nn.Module):
    """Exposes forward hooks for caching intermediate activations."""
    def __init__(self):
        super().__init__()
        self.fwd_hooks = []

    def give_name(self, name):
        self.name = name

    def add_hook(self, hook):
        def full_hook(module, module_input, module_output):
            return hook(module_output, name=self.name)
        self.fwd_hooks.append(self.register_forward_hook(full_hook))

    def remove_hooks(self):
        for h in self.fwd_hooks:
            h.remove()
        self.fwd_hooks = []

    def forward(self, x):
        return x

class Embed(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.W_E = nn.Parameter(torch.randn(d_model, d_vocab) / np.sqrt(d_model))

    def forward(self, x):
        return torch.einsum('dbp -> bpd', self.W_E[:, x])

class Unembed(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.W_U = nn.Parameter(torch.randn(d_model, d_vocab) / np.sqrt(d_vocab))

    def forward(self, x):
        return x @ self.W_U

class PosEmbed(nn.Module):
    def __init__(self, max_ctx, d_model):
        super().__init__()
        self.W_pos = nn.Parameter(torch.randn(max_ctx, d_model) / np.sqrt(d_model))

    def forward(self, x):
        return x + self.W_pos[:x.shape[-2]]

class Attention(nn.Module):
    def __init__(self, d_model, num_heads, d_head, n_ctx):
        super().__init__()
        self.W_K = nn.Parameter(torch.randn(num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_Q = nn.Parameter(torch.randn(num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_V = nn.Parameter(torch.randn(num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_O = nn.Parameter(torch.randn(d_model, d_head * num_heads) / np.sqrt(d_model))
        self.register_buffer('mask', torch.tril(torch.ones((n_ctx, n_ctx))))
        self.d_head = d_head
        self.hook_attn = HookPoint()

    def forward(self, x):
        k = torch.einsum('ihd,bpd->biph', self.W_K, x)
        q = torch.einsum('ihd,bpd->biph', self.W_Q, x)
        v = torch.einsum('ihd,bpd->biph', self.W_V, x)
        
        attn_scores = torch.einsum('biph,biqh->biqp', k, q)
        attn_scores_masked = torch.tril(attn_scores) - 1e10 * (1 - self.mask[:x.shape[-2], :x.shape[-2]])
        attn_matrix = self.hook_attn(
            F.softmax(attn_scores_masked / np.sqrt(self.d_head), dim=-1)
        )
        z = torch.einsum('biph,biqp->biqh', v, attn_matrix)
        z_flat = z.permute(0, 2, 1, 3).flatten(start_dim=2)
        return torch.einsum('df,bqf->bqd', self.W_O, z_flat)

class MLP(nn.Module):
    def __init__(self, d_model, d_mlp):
        super().__init__()
        self.W_in = nn.Parameter(torch.randn(d_mlp, d_model) / np.sqrt(d_model))
        self.b_in = nn.Parameter(torch.zeros(d_mlp))
        self.W_out = nn.Parameter(torch.randn(d_model, d_mlp) / np.sqrt(d_model))
        self.b_out = nn.Parameter(torch.zeros(d_model))
        self.hook_post = HookPoint()

    def forward(self, x):
        pre = torch.einsum('md,bpd->bpm', self.W_in, x) + self.b_in
        post = self.hook_post(F.relu(pre))
        return torch.einsum('dm,bpm->bpd', self.W_out, post) + self.b_out

class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_mlp, d_head, num_heads, n_ctx):
        super().__init__()
        self.attn = Attention(d_model, num_heads, d_head, n_ctx)
        self.mlp = MLP(d_model, d_mlp)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x

class Transformer(nn.Module):
    def __init__(self, num_layers=1, d_vocab=60, d_model=128, d_mlp=512, num_heads=4, n_ctx=3):
        super().__init__()
        self.embed = Embed(d_vocab, d_model)
        self.pos_embed = PosEmbed(n_ctx, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, d_mlp, d_model // num_heads, num_heads, n_ctx)
            for _ in range(num_layers)
        ])
        self.unembed = Unembed(d_vocab, d_model)
        
        for name, module in self.named_modules():
            if isinstance(module, HookPoint):
                module.give_name(name)

    def forward(self, x):
        x = self.embed(x)
        x = self.pos_embed(x)
        for block in self.blocks:
            x = block(x)
        return self.unembed(x)

    def hook_points(self):
        return [m for n, m in self.named_modules() if isinstance(m, HookPoint)]

    def remove_all_hooks(self):
        for hp in self.hook_points():
            hp.remove_hooks()

    def cache_all(self, cache):
        def save_hook(tensor, name):
            cache[name] = tensor.detach()
        for hp in self.hook_points():
            hp.add_hook(save_hook)

# ==========================================
# 2. DATA UTILITIES & CONFIG
# ==========================================

@dataclass
class RunConfig:
    p: int = 131  # Modulus p=131 as requested by the user
    frac_train: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 0.5
    num_epochs: int = 15000
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def cross_entropy_high_precision(logits, labels):
    logprobs = F.log_softmax(logits.to(torch.float64), dim=-1)
    prediction_logprobs = torch.gather(logprobs, index=labels[:, None], dim=-1)
    return -torch.mean(prediction_logprobs)

def gen_split(p, frac, seed, task_name='add'):
    random.seed(seed)
    pairs = [(i, j) for i in range(p) for j in range(p)]
    random.shuffle(pairs)
    
    # Operators mapping
    if task_name == 'add':
        fn = lambda x, y: (x + y) % p
    elif task_name == 'mult':
        fn = lambda x, y: (x * y) % p
    else:
        raise ValueError(f"Unknown task: {task_name}")

    div = int(frac * len(pairs))
    train_pairs, test_pairs = pairs[:div], pairs[div:]
    
    # Build batch tokens: [a, b, op_token] where op_token = p
    train_data = torch.tensor([[a, b, p] for a, b in train_pairs])
    train_labels = torch.tensor([fn(a, b) for a, b in train_pairs])
    test_data = torch.tensor([[a, b, p] for a, b in test_pairs])
    test_labels = torch.tensor([fn(a, b) for a, b in test_pairs])
    
    return train_data, train_labels, test_data, test_labels

# ==========================================
# 3. ANALYSIS METRICS (Tasks 2 & 3)
# ==========================================

def get_fourier_basis(p):
    """Generate 1D Fourier basis terms over Z_p"""
    basis = []
    basis.append(torch.ones(p) / np.sqrt(p))
    for i in range(1, p // 2 + 1):
        cos_term = torch.cos(2 * torch.pi * torch.arange(p) * i / p)
        sin_term = torch.sin(2 * torch.pi * torch.arange(p) * i / p)
        basis.append(cos_term / cos_term.norm())
        basis.append(sin_term / sin_term.norm())
    return torch.stack(basis, dim=0)

def compute_1d_dft(W, basis):
    """Transform embedding matrix W of shape [d_model, p] to Fourier basis"""
    return W @ basis.T

# ==========================================
# 4. SCARCITY SWEEPS (Task 4)
# ==========================================

def run_scarcity_sweep(config):
    fractions = [0.1, 0.3, 0.6, 0.9]
    generalization_epochs = {}
    
    print("\n--- Running Scarcity Sweep (Task 4) ---")
    for frac in fractions:
        torch.manual_seed(config.seed)
        model = Transformer(d_vocab=config.p+1).to(config.device)
        optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        
        train_data, train_labels, test_data, test_labels = gen_split(config.p, frac, config.seed)
        train_data, train_labels = train_data.to(config.device), train_labels.to(config.device)
        test_data, test_labels = test_data.to(config.device), test_labels.to(config.device)
        
        generalized_at = -1
        for epoch in range(1, 5001):  # Fast cap for sweep
            model.train()
            optimizer.zero_grad()
            logits = model(train_data)[:, -1]
            loss = cross_entropy_high_precision(logits, train_labels)
            loss.backward()
            optimizer.step()
            
            # Check validation accuracy
            if epoch % 50 == 0:
                model.eval()
                with torch.no_grad():
                    test_logits = model(test_data)[:, -1]
                    preds = test_logits.argmax(dim=-1)
                    val_acc = (preds == test_labels).float().mean().item()
                    if val_acc > 0.98 and generalized_at == -1:
                        generalized_at = epoch
                        break
        
        generalization_epochs[frac] = generalized_at if generalized_at != -1 else ">5000 (Locked)"
        print(f"Train Fraction: {frac*100:.0f}% | Epochs to Generalize: {generalization_epochs[frac]}")
        
    return generalization_epochs

# ==========================================
# 5. CO-GROKKING (Task 7)
# ==========================================

def run_co_grokking(config):
    """Trains a multi-task network on both modular addition and multiplication"""
    print("\n--- Running Multi-Task Co-Grokking (Task 7) ---")
    
    # We add 2 operator tokens: addition=p, multiplication=p+1
    d_vocab = config.p + 2 
    model = Transformer(d_vocab=d_vocab).to(config.device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    # Generate splits
    add_train_x, add_train_y, add_test_x, add_test_y = gen_split(config.p, config.frac_train, config.seed, 'add')
    mult_train_x, mult_train_y, mult_test_x, mult_test_y = gen_split(config.p, config.frac_train, config.seed, 'mult')
    
    # Re-map multiplication op token to p+1
    mult_train_x[:, 2] = config.p + 1
    mult_test_x[:, 2] = config.p + 1
    
    # Combine datasets
    train_data = torch.cat([add_train_x, mult_train_x], dim=0).to(config.device)
    train_labels = torch.cat([add_train_y, mult_train_y], dim=0).to(config.device)
    test_data = torch.cat([add_test_x, mult_test_x], dim=0).to(config.device)
    test_labels = torch.cat([add_test_y, mult_test_y], dim=0).to(config.device)
    
    history = []
    
    for epoch in range(1, 4001):
        model.train()
        optimizer.zero_grad()
        logits = model(train_data)[:, -1]
        loss = cross_entropy_high_precision(logits, train_labels)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                test_logits = model(test_data)[:, -1]
                preds = test_logits.argmax(dim=-1)
                
                # Separate task validation accuracy
                add_len = len(add_test_y)
                add_acc = (preds[:add_len] == test_labels[:add_len]).float().mean().item()
                mult_acc = (preds[add_len:] == test_labels[add_len:]).float().mean().item()
                
                history.append((epoch, loss.item(), add_acc, mult_acc))
                print(f"Epoch {epoch:>5d} | Train Loss {loss.item():.4f} | Add Val Acc {add_acc*100:.1f}% | Mult Val Acc {mult_acc*100:.1f}%")
                
                if add_acc > 0.98 and mult_acc > 0.98:
                    print(f"✓ Co-grokking achieved successfully at epoch {epoch}!")
                    break

# ==========================================
# 6. ORCHESTRATION PIPELINE
# ==========================================

def main():
    config = RunConfig()
    print(f"Using Modulus p = {config.p} on device: {config.device}")
    
    # Seed verification
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Task 1 & 3: Run full training with phase tracking
    print("\n--- Training Addition Grokking Network (Tasks 1 & 3) ---")
    model = Transformer(d_vocab=config.p+1).to(config.device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    train_data, train_labels, test_data, test_labels = gen_split(config.p, config.frac_train, config.seed)
    train_data, train_labels = train_data.to(config.device), train_labels.to(config.device)
    test_data, test_labels = test_data.to(config.device), test_labels.to(config.device)
    
    train_losses = []
    test_losses = []
    
    for epoch in range(1, 6001):
        model.train()
        optimizer.zero_grad()
        logits = model(train_data)[:, -1]
        loss = cross_entropy_high_precision(logits, train_labels)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        if epoch % 200 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                test_logits = model(test_data)[:, -1]
                test_loss = cross_entropy_high_precision(test_logits, test_labels)
                test_losses.append(test_loss.item())
                preds = test_logits.argmax(dim=-1)
                acc = (preds == test_labels).float().mean().item()
                print(f"Epoch {epoch:>5d} | Train Loss {loss.item():.4f} | Val Loss {test_loss.item():.4f} | Val Acc {acc*100:.1f}%")
                
                if acc > 0.99:
                    print(f"✓ Addition Grokking phase transition completed at epoch {epoch}!")
                    break
                    
    # Task 2: DFT Extract
    print("\n--- Extracting Sparse Fourier Components (Task 2) ---")
    basis = get_fourier_basis(config.p).to(config.device)
    # Remove operator token projection
    W_E = model.embed.W_E[:, :-1] 
    W_Ef = compute_1d_dft(W_E, basis)
    norms = W_Ef.norm(dim=0).cpu().detach().numpy()
    
    # Identify top key frequencies
    top_indices = np.argsort(norms)[-6:]
    print("Detected sparse structural Fourier components (frequencies w_k):")
    for idx in top_indices:
        if idx > 0:
            freq = (idx + 1) // 2
            term = "cos" if idx % 2 != 0 else "sin"
            print(f"  Component: {term}({freq}) | Spectral Norm: {norms[idx]:.4f}")

    # Run remaining demonstration sweeps
    run_scarcity_sweep(config)
    run_co_grokking(config)

if __name__ == "__main__":
    main()
