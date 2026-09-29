import time
from typing import Protocol
from collections import defaultdict

from jax import Array

from .evaluation import mse
from .utils import update_results_dict


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.elapsed = time.perf_counter() - self.start


class GBTrackerLike(Protocol):
    """Protocol for GB training tracker callbacks."""

    def track(self, params: list[Array], train_time: float) -> None:
        """Log or track training progress."""


class GBTracker:
    def __init__(
        self, 
        x_tr,
        y_tr,
        x_val,
        y_val,
        pred_f,
        gamma_w
    ):
        self.x_tr = x_tr
        self.y_tr = y_tr
        self.x_val = x_val
        self.y_val = y_val
        self.pred_f = pred_f
        self.res_dict = defaultdict(list)
        self._track_count = 0
        self.gamma_w = gamma_w
 
    def track(self, params: list[Array], train_time: float) -> None:
        self._track_count += 1
        y_pred = self.pred_f(params, self.x_val)
        update_results_dict(
            self.res_dict,
            est_n=self._track_count,
            train_time_v=train_time,
            mse_v=mse(self.y_val, y_pred).item(),
            mse_tr=mse(self.y_tr, self.pred_f(params, self.x_tr)).item(),
            gamma_w=self.gamma_w
        )


class Tracker:
    def __init__(
        self, 
        x_val,
        y_val,
    ):
        self.x_val = x_val
        self.y_val = y_val
        self.res_dict = defaultdict(list)
 
    def track(self, rank, gamma_w, model, train_info):
        update_results_dict(
            self.res_dict, 
            rank_v=rank,
            gamma_w=gamma_w,
            train_time_v=train_info['train_time'],
            mse_v=mse(self.y_val, model.predict(self.x_val)).item(),
        )
