from abc import ABCMeta, abstractmethod
import random


class Layer(metaclass=ABCMeta):
    @abstractmethod
    def compute_input_errors(self, output_errors):
        """Checks how the input to the layer should change to improve the output - reduce output_errors

        The result of this function tells us how the input (the output of the previous layer) should be modified to reduce the output errors.

        Args:
            output_errors: errors for each output of the layer calculated in the next layer.

        Returns:
            A list of input errors - this tells the previous layer how it should be modified to improve the output
        """
    
    @abstractmethod
    def compute_gradients(self, input_, output_errors):
        """Compute gradients over the parameters. This tells how much each parameter affects the output_errors."""
    
    @abstractmethod
    def update_params(self, gradients, learning_rate):
        """Update parameters of the layer."""
    
    @abstractmethod
    def __call__(self, input_):
        """"We think of a layer as a function that process input in a certain way."""
 
 
class LinearLayer(Layer):
    def __init__(self, input_features: int, output_features: int):
        self.n_input_features = input_features
        self.n_output_features = output_features
        self.weights = []
        self.biases = []
        for j in range(input_features):
            W_j = [] # weights for j-th feature
            for k in range(output_features):
                W_jk = random.uniform(0, 0.1) # weight that represents j-th feature affects k-th neuron 
                W_j.append(W_jk)
            self.weights.append(W_j)
        for k in range(output_features):
            bias_k = 0#random.uniform(0, 1) # number that should be added to the k-th neuron
            self.biases.append(bias_k)
            
    def compute_input_errors(self, output_errors):
        """Check how the input to the layer should change to provide best output (reduce output errors to 0)"""
        assert len(output_errors) == self.n_output_features
        input_errors = []
        for j in range(self.n_input_features):
            grad_j = 0
            for k in range(self.n_output_features):
                grad_j += output_errors[k] * self.weights[j][k]
            input_errors.append(grad_j)
        return input_errors

    def compute_gradients(self, input_, output_errors):
        """Compute gradients over the parameters. This tells how much each parameter affects the output_errors."""
        assert len(output_errors) == self.n_output_features
        dW = []
        db = []
        for j in range(self.n_input_features):
            dW_j = []
            for k in range(self.n_output_features):
                dW_jk = output_errors[k] * input_[j]
                dW_j.append(dW_jk)
            dW.append(dW_j)
        for k in range(self.n_output_features):
            db_k = output_errors[k]
            db.append(db_k)
        return dW, db # I can shift on my own - it doesn't depend on the input!

    def update_params(self, gradients, learning_rate):
        # gradients: (dW, db)
        dW = gradients[0]
        db = gradients[1]
        for j in range(self.n_input_features):
            for k in range(self.n_output_features):
                self.weights[j][k] -= dW[j][k] * learning_rate
        for k in range(self.n_output_features):
            self.biases[k] -= db[k] * learning_rate
        
    def __call__(self, input_):
        output = []
        for k in range(self.n_output_features):
            output_k = 0
            for j in range(self.n_input_features):
                output_k += self.weights[j][k] * input_[j]
            # add bias 
            output_k += self.biases[k]
            output.append(output_k)
        return output
