"""Training routines for NN and PINN models."""

import jax
import jax.numpy as jnp
from jax import jit
from tqdm import tqdm

from .config import Config
from .loss import bc_loss, data_loss, ic_loss, physics_loss
from .model import init_nn_params, init_pinn_params
from .optim import adam_step, init_adam
from .sampling import sample_bc, sample_ic, sample_interior


def train_nn(
    sensor_data: jnp.ndarray, cfg: Config
) -> tuple[list[tuple[jnp.ndarray, jnp.ndarray]], dict]:
    """Train a standard neural network on sensor data only.

    Args:
        sensor_data: Sensor measurements [x, y, t, T]
        cfg: Configuration

    Returns:
        params: Trained network parameters
        losses: Dictionary of loss histories
    """
    key = jax.random.key(cfg.seed)
    nn_params = init_nn_params(cfg) #Lager liste av weights og biases
    adam_state = init_adam(nn_params) # initialiserer starttilstanden

    losses = {"total": [], "data": [], "ic": []}  # Fill with loss histories

    #######################################################################
    # Oppgave 4.3: Start
    #######################################################################

    #Definerer objektivfunksjon, det er dette vi ønsker å minimere og korrigerer våre weights og biaser etter. 
    def objective_fn(nn_params, ic_punkt): 

        l_data = data_loss(nn_params, sensor_data, cfg)
        l_ic = ic_loss(nn_params, ic_punkt, cfg)
        l_total = cfg.lambda_data * l_data + cfg.lambda_ic * l_ic

        return l_total, (l_data, l_ic)

    #Definer funksjon for hvert "step", som korrigerer NN parametrene basert på objektivfunksjonen og lagrer verdiene for total loss, og idividuell loss for data og initialbetingelse
    @jit # bruker Jax for å kjøre fortere...
    def step(nn_params, adam_state, ic_punkt): 
        (loss, (l_data, l_ic)), grads = jax.value_and_grad(objective_fn, has_aux=True)(nn_params, ic_punkt)

        nn_params, adam_state = adam_step(nn_params, grads, adam_state, lr=cfg.learning_rate) #oppdaterer nye parametre og nytt adam state

        return nn_params, adam_state, loss, l_data, l_ic

    # itererer over antall epoker, med progress bar som oppgitt i oppgaven, itererer ved bruk av step funksjonen vi definerte ovenfor.
    for _ in tqdm(range(cfg.num_epochs), desc="Training NN"): 
        ic_punkt, key = sample_ic(key, cfg) 

        nn_params, adam_state, l_tot, l_data, l_ic = step(nn_params, adam_state, ic_punkt)  #Oppdaterer NN parametrene

        # lagrer losses for plotting
        losses["total"].append(l_tot)
        losses["data"].append(l_data)
        losses["ic"].append(l_ic)

    #######################################################################
    # Oppgave 4.3: Slutt
    #######################################################################

    return nn_params, {k: jnp.array(v) for k, v in losses.items()}


def train_pinn(sensor_data: jnp.ndarray, cfg: Config) -> tuple[dict, dict]:
    """Train a physics-informed neural network.

    Args:
        sensor_data: Sensor measurements [x, y, t, T]
        cfg: Configuration

    Returns:
        pinn_params: Trained parameters (nn weights + alpha)
        losses: Dictionary of loss histories
    """
    key = jax.random.key(cfg.seed)
    pinn_params = init_pinn_params(cfg)
    opt_state = init_adam(pinn_params)

    losses = {"total": [], "data": [], "physics": [], "ic": [], "bc": []}

    #######################################################################
    # Oppgave 5.3: Start
    #######################################################################
    @jax.jit #Jit for raskere derivasjon og derfor programkjøring
    def step(pinn_params, opt_state, interior_epoch, ic_epoch, bc_epoch): 
        def objective_fn(pinn_params,interior_epoch,ic_epoch,bc_epoch): #Etablerer ny objektivfunksjon for hvert steg 
            l_data = data_loss(pinn_params["nn"], sensor_data, cfg)
            l_ic = ic_loss(pinn_params["nn"], ic_epoch, cfg)
            l_physics = physics_loss(pinn_params, interior_epoch, cfg)
            l_bc = bc_loss(pinn_params, bc_epoch, cfg)

            l_total = (
            cfg.lambda_data * l_data
            + cfg.lambda_ic * l_ic
            + cfg.lambda_physics * l_physics
            + cfg.lambda_bc * l_bc
        )
            return l_total, (l_data, l_ic, l_physics, l_bc)
        
        (loss, (l_data, l_ic, l_physics, l_bc)), grads = jax.value_and_grad(
            objective_fn, has_aux=True
        )(pinn_params, interior_epoch, ic_epoch, bc_epoch)

        pinn_params, opt_state = adam_step(pinn_params, grads, opt_state, lr=cfg.learning_rate) #Oppdaterer PINN parametre
        return pinn_params, opt_state, loss, l_data, l_ic, l_physics, l_bc


    for _ in tqdm(range(cfg.num_epochs), desc="Training PINN"):
        interior_epoch, key = sample_interior(key, cfg)
        ic_epoch, key = sample_ic(key, cfg)
        bc_epoch, key = sample_bc(key, cfg)

        pinn_params, opt_state, l_tot, l_data, l_ic, l_physics, l_bc = step(
            pinn_params, opt_state, interior_epoch, ic_epoch, bc_epoch
        )

        losses["total"].append(l_tot)
        losses["data"].append(l_data)
        losses["ic"].append(l_ic)
        losses["physics"].append(l_physics)
        losses["bc"].append(l_bc)

    #######################################################################
    # Oppgave 5.3: Slutt
    #######################################################################

    return pinn_params, {k: jnp.array(v) for k, v in losses.items()}
