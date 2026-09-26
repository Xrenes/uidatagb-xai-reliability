"""
server.py
---------
Flower Federated Learning Server.

Orchestrates federated rounds:
  1. Sends global model weights to all clients
  2. Clients train locally and return updated weights
  3. Server aggregates weights using FedAvg or FedProx strategy
  4. Repeat for N rounds

Run this FIRST before starting any clients:
    python server.py --rounds 10 --strategy fedavg
    python server.py --rounds 10 --strategy fedprox --proximal_mu 0.1
"""

import argparse
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import flwr as fl
from flwr.common import (
    FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg, FedProx

import torch
from model import build_model, set_model_parameters, get_model_parameters

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "./outputs/checkpoints"


# ── Custom FedAvg with checkpoint saving ─────────────────────────────────────
class FedAvgWithCheckpoint(FedAvg):
    """
    Extends FedAvg to save global model weights after each round.
    Also tracks aggregated metrics (loss, accuracy) across rounds.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.round_metrics: List[Dict] = []
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ):
        aggregated_params, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_params is not None:
            # Save checkpoint
            ndarrays = parameters_to_ndarrays(aggregated_params)
            model = build_model(pretrained=False).to(DEVICE)
            set_model_parameters(model, ndarrays)
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"round_{server_round:03d}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"[Server] Round {server_round} checkpoint saved → {ckpt_path}")

        return aggregated_params, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results,
        failures,
    ):
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )

        if aggregated_loss is not None:
            acc = aggregated_metrics.get("accuracy", 0)
            print(f"[Server] Round {server_round} | "
                  f"Aggregated Loss: {aggregated_loss:.4f} | "
                  f"Aggregated Accuracy: {acc:.4f}")
            self.round_metrics.append({
                "round": server_round,
                "loss": aggregated_loss,
                "accuracy": acc,
            })

        return aggregated_loss, aggregated_metrics


# ── Initial model parameters ──────────────────────────────────────────────────
def get_initial_parameters() -> Parameters:
    model = build_model(pretrained=True)
    ndarrays = get_model_parameters(model)
    return ndarrays_to_parameters(ndarrays)


# ── Entry Point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10,
                        help="Number of federated learning rounds")
    parser.add_argument("--num_clients", type=int, default=3,
                        help="Minimum number of clients to wait for")
    parser.add_argument("--strategy", type=str, default="fedavg",
                        choices=["fedavg", "fedprox"],
                        help="Aggregation strategy")
    parser.add_argument("--proximal_mu", type=float, default=0.1,
                        help="FedProx proximal term weight (only used with fedprox)")
    parser.add_argument("--server_address", type=str, default="0.0.0.0:8080")
    args = parser.parse_args()

    print(f"[Server] Strategy : {args.strategy.upper()}")
    print(f"[Server] Rounds   : {args.rounds}")
    print(f"[Server] Clients  : {args.num_clients}")

    # Shared config sent to every client each round
    def fit_config(server_round: int) -> Dict[str, Scalar]:
        return {
            "local_epochs": 3,
            "round": server_round,
        }

    initial_params = get_initial_parameters()

    if args.strategy == "fedprox":
        strategy = FedProx(
            proximal_mu=args.proximal_mu,
            min_fit_clients=args.num_clients,
            min_evaluate_clients=args.num_clients,
            min_available_clients=args.num_clients,
            on_fit_config_fn=fit_config,
            initial_parameters=initial_params,
        )
        print(f"[Server] FedProx proximal_mu = {args.proximal_mu}")
    else:
        strategy = FedAvgWithCheckpoint(
            min_fit_clients=args.num_clients,
            min_evaluate_clients=args.num_clients,
            min_available_clients=args.num_clients,
            on_fit_config_fn=fit_config,
            initial_parameters=initial_params,
        )

    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
