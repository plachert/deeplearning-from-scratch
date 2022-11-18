from abc import ABCMeta, abstractmethod


class Loss(metaclass=ABCMeta):
    
    @abstractmethod
    def compute_loss_grad(self, y_hat, y):
        """Computes dL/dy_hat"""
    
    @abstractmethod
    def compute_cost(self, y_hat, y):
        """Computes loss."""


class MSELoss:
    def compute_loss_grad(self, y_hat, y):
        return [2 * (y_hat[0] - y[0])]

    def compute_cost(self, output, y):
        return (y[0] - y_hat[0]) ** 2