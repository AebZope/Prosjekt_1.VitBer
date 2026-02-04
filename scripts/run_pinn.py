"""Script for training and plotting the PINN model."""

import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from viz import create_animation, plot_snapshots

from project import (
    generate_training_data,
    load_config,
    predict_grid,
    train_pinn,
)


def main():
    cfg = load_config("config.yaml")

    #######################################################################
    # Oppgave 5.4: Start
    #######################################################################
    print("Solving heat equation with PINN...")

    # Generer treningsdata
    x, y, t, T_fdm, sensor_data = generate_training_data(cfg)

    pinn_params, losses = train_pinn(sensor_data, cfg)

    # Prediksjon på grid
    T_pred = predict_grid(pinn_params['nn'], x, y, t, cfg) 

    # Plot treningsloss
    epochs = np.arange(cfg.num_epochs)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses["total"], label="Total loss")
    plt.plot(epochs, losses["data"], label="Data loss")
    plt.plot(epochs, losses["physics"], label="Physics loss")
    plt.plot(epochs, losses["ic"], label="IC loss")
    plt.plot(epochs, losses["bc"], label="BC loss")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("PINN training losses")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

     # Lag animasjon og snapshots
    os.makedirs("output/pinn", exist_ok=True)
    create_animation(
        x, y, t, T_pred, title="PINN", save_path="output/pinn/pinn_animation.gif"
    )

    plot_snapshots(
        x,
        y,
        t,
        T_pred,
        save_path="output/pinn/pinn_snapshots.png",
    )

    # Print lærte fysiske parametere
    alpha = jnp.exp(pinn_params["log_alpha"])
    power = jnp.exp(pinn_params["log_power"])
    k     = jnp.exp(pinn_params["log_k"])
    h     = jnp.exp(pinn_params["log_h"])

    print("\nLearned physical parameters:")
    print(f"alpha = {alpha}")
    print(f"power = {power}")
    print(f"k     = {k}")
    print(f"h     = {h}")

    #######################################################################
    # Oppgave 5.4: Slutt
    #######################################################################


if __name__ == "__main__":
    main()
