from typing import List
from layer import Layer
from loss import Loss

class Model:
    def __init__(self, layers: List[Layer], loss: Loss, learning_rate: float = 0.001):
        self.layers = layers
        self.loss = loss
        self.learning_rate = learning_rate
        self.outputs = [] # stores activations after forward pass
    
    def forward_pass(self, input_):
        self.outputs = [] # reset
        self.outputs.append(input_) # treat input features as the first activation
        for layer in self.layers:
            output = layer(input_)
            self.outputs.append(output)
            input_ = output
        return output

    def backpropagation_step(self, target):
        output_errors = self.loss.compute_loss_grad(self.outputs[-1], target)
        layers_grads = []
        for i, (layer, output) in enumerate(zip(self.layers[::-1], self.outputs[::-1])):
            input_errors = layer.compute_input_errors(output_errors)
            input_ = self.outputs[::-1][i+1] # output of the previous layer
            grads = layer.compute_gradients(input_, output_errors)
            output_errors = input_errors
            layers_grads.append((layer, grads))
        return layers_grads
    
    def update_layers(self, layers_grads):
        for layer, grads in layers_grads:
            layer.update_params(grads, self.learning_rate)
    
    def fit(self, X, y, epochs=10):
        for epoch in range(epochs):
            loss_avg = 0
            for X_, y_ in zip(X, y):
                y_hat = self.forward_pass(X_)
                layers_grads = self.backpropagation_step(y_)
                self.update_layers(layers_grads)
                loss = self.loss.compute_cost(y_hat, y_)
                loss_avg += loss
            loss_avg /= len(y)
            print(f"Loss after epoch {epoch}: {loss_avg}")

if __name__ == "__main__":
    features = [0.2, 0.8]#np.random.random((5, 10))
    y = [1]
    model = Model(layers=[LinearLayer(2, 1)], loss=MSELoss(), learning_rate=0.0001)
    model.fit(features, y)
    print(model.forward_pass(features))
    print(model.layers[0].weights, model.layers[0].biases)