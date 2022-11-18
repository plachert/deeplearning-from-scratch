from model import Model
from layer import LinearLayer
from loss import MSELoss
import random


def generate_simple_regression(n_samples=1000, a=2, b=-1, x_lim=[-10, 10]):
    linear_func = lambda x: a*x + b + random.uniform(-1, 1) # y = ax + b + noise
    X = []
    y = []
    for _ in range(n_samples):
        X_ = random.uniform(*x_lim)
        y_ = linear_func(X_)
        X.append([X_])
        y.append([y_])
    return X, y 
    

def main():
    layers = [LinearLayer(1, 1)]
    loss = MSELoss()
    model = Model(layers=layers, loss=loss, learning_rate=0.001)
    X, y = generate_simple_regression()
    model.fit(X, y)
    print(model.layers[0].weights, model.layers[0].biases)
    

if __name__ == "__main__":
    main()