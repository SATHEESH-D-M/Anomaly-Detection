import logging


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, verbose=False, mode="max"):
        """
        Early stopping for classification tasks based on a monitored metric.

        Args:
            patience (int): How many epochs to wait after last improvement.
            min_delta (float): Minimum change to qualify as an improvement.
            verbose (bool): Whether to log messages.
            mode (str): 'max' for metrics to maximize (F1, accuracy), 'min' for loss.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.verbose = verbose
        self.best_model_state = None
        self.mode = mode

    def __call__(self, metric, model):
        improved = False
        if self.best_score is None:
            improved = True
        elif self.mode == "max" and metric > self.best_score + self.min_delta:
            improved = True
        elif self.mode == "min" and metric < self.best_score - self.min_delta:
            improved = True

        if improved:
            self.best_score = metric
            self.counter = 0
            self.best_model_state = model.state_dict()
            if self.verbose:
                logging.info(f"Monitored metric improved to {metric:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                logging.info(
                    f"Monitored metric did not improve. Counter: {self.counter}/{self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logging.warning("Early stopping triggered.")
