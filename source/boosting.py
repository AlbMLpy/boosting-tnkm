from functools import partial

import jax
from jax import jit
from jax import Array
import jax.numpy as jnp
from tnkm.models.base import TNKM
from tnkm.core.cpkm import predict_jit
from tnkm.features import ProductFeatures
from tnkm.optim.als import als_step_cp, prepare_buffer_cp

from source.tracker import Timer, GBTrackerLike


DIV_EPS = 1e-14 # Prevent division by zero


@partial(jit, static_argnames=("fmap", "n_epoch", "d_core_stop", "reg_mode"))
def _als_cp_jax(
    params: list[Array], 
    x: Array, 
    y: Array, 
    fmap: ProductFeatures, 
    n_epoch: int, 
    gamma_w: float, 
    beta_e: float, 
    d_core_stop: int | None, 
    reg_mode: str,
) -> list[Array]:
    """Several ALS sweeps: update all CPD cores sequentially."""
    buf = prepare_buffer_cp(params, fmap, x)
    for _ in range(n_epoch):
        params, buf = als_step_cp(
            params, fmap, x, y, gamma_w, 
            beta_e, buf, d_core_stop, reg_mode,
        )
    return params


def _gb_train_cp(
    params_init: list[Array], 
    x: Array, 
    y: Array, 
    fmap: ProductFeatures, 
    n_epoch: int, 
    gamma_w: float, 
    beta_e: float, 
    d_core_stop: int | None, 
    reg_mode: str,
    tracker: GBTrackerLike | None,
) -> list[Array]:
    """Gradient boosting training of CPKM."""
    for m in range(len(params_init)):
        with Timer() as timer:
            if m == 0: # Train the first estimator
                params = _als_cp_jax(
                    params_init[m], x, y, fmap, n_epoch, 
                    gamma_w, beta_e, d_core_stop, reg_mode
                )
                y_hat = predict_jit(params, x, fmap)
            else:
                # Compute m-step pseudo-residual
                r_err = y - y_hat
                # Compute optimal m-estimator model
                pm = _als_cp_jax(
                    params_init[m], x, r_err, fmap, n_epoch, 
                    gamma_w, beta_e, d_core_stop, reg_mode,
                )
                # Compute optimal step-size
                pm_pred = predict_jit(pm, x, fmap)
                beta_m = 1 # For simplicity
                # Update the model
                y_hat = y_hat + beta_m*pm_pred
                _concat = lambda x, y: jnp.concatenate((x, beta_m*y), axis=1)
                params = list(map(_concat, params, pm))
            jax.block_until_ready(params[0])
        if tracker is not None:
            tracker.track(params, timer.elapsed)
    return params


def gb_train(
    model: TNKM, 
    x: Array, 
    y: Array, 
    n_estimators: int, 
    n_epoch: int,
    gamma_w: float, 
    beta_e: float, 
    d_core_stop: int | None = None, 
    reg_mode: str = "cores",
    tracker: GBTrackerLike | None = None,
) -> TNKM:  
    """Train a TNKM model using Gradient Boosting (GB) and ALS.
    
    ALS optimizes each base learner (model) by sequentially updating one
    core at a time while keeping all others fixed. After that, the GB update 
    formulas are applied to update the final model.

    Parameters
    ----------
    model : TNKM
        Base learner as a tensor network kernel machine. CP-based as of now.
    x : Array
        Input data of shape (n_samples, d_dim).
    y : Array
        Target values of shape (n_samples,).
    n_estimators: int
        The number of boosting stages to perform.
    n_epoch : int
        Number of training epochs. Must be positive.
    gamma_w : float, default=0.0
        Weight of the L2 regularization term.
    beta_e : float, default=1.0
        Weight of the mean squared error term.
    d_core_stop : int | None, default=None
        If provided, the final epoch updates only cores with indices
        smaller or equal to ``d_core_stop``. This can be used to
        freeze the remaining cores during the final sweep.
    reg_mode: {"cores", "tensor"}, default="cores"
        Regularization strategy.
        - ``"cores"``: apply L2 regularization directly to the tensor cores.
        - ``"tensor"``: apply L2 regularization to the reconstructed tensor.
    tracker : GBTrackerLike | None, default=None
        Optional training callback invoked before training
        and after each epoch.

    Returns
    -------
    TNKM
        A fully trained GB-TNKM model.
    """
    model_cls, fmap = type(model), model.fmap
    m_seed, dtype = model.seed, model.dtype
    params_init = [
        model_cls(fmap, model.rank, seed, dtype).params 
        for seed in range(m_seed, m_seed + n_estimators)
    ]
    params = _gb_train_cp(
        params_init, x, y, fmap, n_epoch, gamma_w, 
        beta_e, d_core_stop, reg_mode, tracker,
    )
    model = model_cls(fmap, params[0].shape[1], dtype=dtype)
    model.params = params
    return model
