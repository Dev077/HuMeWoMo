import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from HuMeWoMo.datasets.homo_binding_dataset import get_homo_dataloaders
from HuMeWoMo.models.homo_binding_model import HomoBindingModel

# =============================================================================
# HYPERPARAMETERS & CONFIG
# =============================================================================
DATA_DIR = "./bindingdb_data/final_dataset"
BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Save Constants
MIN_IMPROVEMENT = 0.005  # Save backup if val loss improves by at least this much
MODEL_SAVE_PATH = "best_homo_model.pt"
BACKUP_SAVE_PATH = "best_homo_model_backup.pt"

# Plotting
FIGS_DIR = os.path.join(project_root, "figs")
os.makedirs(FIGS_DIR, exist_ok=True)

# =============================================================================
# TRAINING LOOP
# =============================================================================
def train():
    print(f"Training on device: {DEVICE}")

    # Load Data
    train_loader, val_loader, test_loader = get_homo_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=4 if DEVICE.type == "cuda" else 0
    )

    # Initialize Model
    model = HomoBindingModel(
        drug_in_dim=50,
        enzyme_in_dim=27,
        hidden_dim=128,
        n_heads=4,
        num_drug_layers=3,
        num_enzyme_layers=3,
        num_decoder_layers=3
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'test_loss': []}

    for epoch in range(1, EPOCHS + 1):
        # --- TRAIN ---
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]")
        
        for batch in pbar:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            
            preds = model(batch)
            loss = criterion(preds, batch.y)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # --- VALIDATE ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                preds = model(batch)
                val_loss += criterion(preds, batch.y).item()
        
        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        
        # --- TEST (Track only) ---
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(DEVICE)
                preds = model(batch)
                test_loss += criterion(preds, batch.y).item()
        
        avg_test_loss = test_loss / len(test_loader)
        history['test_loss'].append(avg_test_loss)

        print(f"  Summary: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Test Loss: {avg_test_loss:.4f}")

        # --- SCHEDULER & SAVING ---
        scheduler.step(avg_val_loss)

        if avg_val_loss < (best_val_loss - MIN_IMPROVEMENT):
            print(f"  Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}. Saving backup...")
            best_val_loss = avg_val_loss
            # Save the primary best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, MODEL_SAVE_PATH)
            
            # Save a secondary backup
            torch.save(model.state_dict(), BACKUP_SAVE_PATH)

    # --- FINAL EVALUATION ON VAL ---
    print("\n" + "="*50)
    print("FINAL VALIDATION RUN")
    print("="*50)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH)['model_state_dict'])
    model.eval()
    final_val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Final Val"):
            batch = batch.to(DEVICE)
            preds = model(batch)
            final_val_loss += criterion(preds, batch.y).item()
    print(f"Final Validation Loss (MSE): {final_val_loss / len(val_loader):.4f}")

    # --- PLOT LOSS ---
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('HomoBindingModel Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(FIGS_DIR, "training_loss.png"))
    print(f"Loss plot saved to {FIGS_DIR}/training_loss.png")

if __name__ == "__main__":
    train()
