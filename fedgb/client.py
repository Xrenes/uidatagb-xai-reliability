"""
client.py
---------
Flower Federated Learning Client.

Each client represents one simulated hospital.
It trains the local model on its private data partition
and reports weights back to the FL server — never the raw images.

Usage (run in separate terminals, one per client):
    python client.py --client_id 0 --data_dir ./data --num_clients 3
    python client.py --client_id 1 --data_dir ./data --num_clients 3
    python client.py --client_id 2 --data_dir ./data --num_clients 3
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import flwr as fl

from dataset import get_client_datasets
from model import build_model, get_model_parameters, set_model_parameters

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Local Training ────────────────────────────────────────────────────────────
def train(model, loader, optimizer, criterion, epochs: int = 1):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for _ in range(epochs):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / len(loader)
    return avg_loss, accuracy


# ── Local Evaluation ──────────────────────────────────────────────────────────
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / len(loader)
    return avg_loss, accuracy


# ── Flower Client Class ───────────────────────────────────────────────────────
class GBClient(fl.client.NumPyClient):

    def __init__(self, client_id: int, train_loader, val_loader):
        self.client_id = client_id
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.model = build_model(pretrained=True).to(DEVICE)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=1e-4, weight_decay=1e-4
        )

    def get_parameters(self, config):
        print(f"[Client {self.client_id}] Sending parameters to server")
        return get_model_parameters(self.model)

    def set_parameters(self, parameters):
        set_model_parameters(self.model, parameters)

    def fit(self, parameters, config):
        """Receive global weights → train locally → send updated weights back."""
        self.set_parameters(parameters)

        local_epochs = int(config.get("local_epochs", 3))
        print(f"[Client {self.client_id}] Training for {local_epochs} epoch(s)...")

        loss, acc = train(
            self.model, self.train_loader,
            self.optimizer, self.criterion,
            epochs=local_epochs
        )
        print(f"[Client {self.client_id}] Train Loss: {loss:.4f} | Acc: {acc:.4f}")

        return get_model_parameters(self.model), len(self.train_loader.dataset), {
            "train_loss": loss,
            "train_accuracy": acc,
        }

    def evaluate(self, parameters, config):
        """Receive global weights → evaluate on local validation set."""
        self.set_parameters(parameters)

        loss, acc = evaluate(self.model, self.val_loader, self.criterion)
        print(f"[Client {self.client_id}] Val Loss: {loss:.4f} | Acc: {acc:.4f}")

        return loss, len(self.val_loader.dataset), {"accuracy": acc}


# ── Entry Point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, required=True,
                        help="Client index (0-based)")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Path to GBCU dataset root")
    parser.add_argument("--num_clients", type=int, default=3,
                        help="Total number of clients")
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080",
                        help="FL server address")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    print(f"[Client {args.client_id}] Using device: {DEVICE}")

    # Load this client's data partition
    client_datasets, _ = get_client_datasets(args.data_dir, args.num_clients)
    client_data = client_datasets[args.client_id]

    # 80/20 train/val split within the client
    n_val = max(1, int(0.2 * len(client_data)))
    n_train = len(client_data) - n_val
    train_set, val_set = random_split(client_data, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    client = GBClient(args.client_id, train_loader, val_loader)

    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=client,
    )


if __name__ == "__main__":
    main()
