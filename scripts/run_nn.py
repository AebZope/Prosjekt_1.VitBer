"""Script for training and plotting the NN model."""

import os

import matplotlib.pyplot as plt
import numpy as np
from viz import create_animation, plot_snapshots

from project import (
    generate_training_data,
    load_config,
    predict_grid,
    train_nn,
)


def main():
    cfg = load_config("config.yaml")

    #######################################################################
    # Oppgave 4.4: Start
    #######################################################################
    print("Solving heat equation with NN...")
    x,y,t,_,training = generate_training_data(cfg) # Produsererer treningsdata som vi bruker for å korrigere parametrene våre. 
    nn_params,losses = train_nn(training,cfg)  #trener parametrene, weight og bias. 

    T_pred = predict_grid(nn_params,x,y,t,cfg) #Bruker de trente parametrene til å predikere en temperatur grid for systemet

    epochs = np.arange(cfg.num_epochs)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses["total"], label="Total loss")
    #plt.plot(epochs, losses["data"], label="Data loss")
    #plt.plot(epochs, losses["ic"], label="IC loss")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("NN training losses")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show() 

    create_animation(
        x, y, t, T_pred, title="FDM", save_path="output/fdm/fdm_animation.gif"
    )

    plot_snapshots(
        x,
        y,
        t,
        T_pred,
        save_path="output/fdm/fdm_snapshots.png",
    )


    #######################################################################
    # Oppgave 4.4: Slutt
    #######################################################################


if __name__ == "__main__":
    main()