import numpy as np
import matplotlib.pyplot as plt

# Activation functions
def sigmoid(x):
    """Sigmoid activation function: f(x) = 1 / (1 + e^(-x))"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow

def deriv_sigmoid(x):
    """Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))"""
    fx = sigmoid(x)
    return fx * (1 - fx)

def relu(x):
    """ReLU activation function: f(x) = max(0, x)"""
    return np.maximum(0, x)

def deriv_relu(x):
    """Derivative of ReLU: f'(x) = 1 if x > 0, else 0"""
    return (x > 0).astype(float)

# Loss functions
def mse_loss(y_true, y_pred):
    """Mean Squared Error loss"""
    return ((y_true - y_pred) ** 2).mean()

def binary_crossentropy(y_true, y_pred):
    """Binary cross-entropy loss"""
    # Clip predictions to prevent log(0)
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

class Neuron:
    """A single neuron with weights, bias, and activation function"""
    
    def __init__(self, weights, bias, activation='sigmoid'):
        self.weights = weights
        self.bias = bias
        self.activation = activation
        
    def feedforward(self, inputs):
        """Forward pass through the neuron"""
        total = np.dot(self.weights, inputs) + self.bias
        
        if self.activation == 'sigmoid':
            return sigmoid(total)
        elif self.activation == 'relu':
            return relu(total)
        else:
            return total  # Linear activation

class NeuralNetwork:
    """
    A flexible neural network implementation
    Can handle multiple hidden layers and different activation functions
    """
    
    def __init__(self, input_size, hidden_sizes, output_size, activation='sigmoid'):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation = activation
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        # Create layers
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        for i in range(len(layer_sizes) - 1):
            # Xavier initialization for better convergence
            limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i + 1]))
            w = np.random.uniform(-limit, limit, (layer_sizes[i + 1], layer_sizes[i]))
            b = np.zeros(layer_sizes[i + 1])
            
            self.weights.append(w)
            self.biases.append(b)
    
    def _activate(self, x, layer_idx):
        """Apply activation function"""
        if layer_idx == len(self.weights) - 1:  # Output layer
            return sigmoid(x)  # Always sigmoid for binary classification
        
        if self.activation == 'sigmoid':
            return sigmoid(x)
        elif self.activation == 'relu':
            return relu(x)
        else:
            return x
    
    def _activate_derivative(self, x, layer_idx):
        """Compute derivative of activation function"""
        if layer_idx == len(self.weights) - 1:  # Output layer
            return deriv_sigmoid(x)
        
        if self.activation == 'sigmoid':
            return deriv_sigmoid(x)
        elif self.activation == 'relu':
            return deriv_relu(x)
        else:
            return np.ones_like(x)
    
    def feedforward(self, x):
        """Forward pass through the network"""
        self.layer_inputs = [x]  # Store for backpropagation
        self.layer_outputs = [x]
        
        current_input = x
        
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(w, current_input) + b
            a = self._activate(z, i)
            
            self.layer_inputs.append(z)
            self.layer_outputs.append(a)
            current_input = a
        
        return current_input
    
    def backpropagate(self, x, y_true, learning_rate=0.1):
        """Backpropagation algorithm"""
        # Forward pass
        y_pred = self.feedforward(x)
        
        # Calculate output layer error
        output_error = 2 * (y_pred - y_true)  # MSE derivative
        deltas = [output_error]
        
        # Backpropagate errors
        for i in range(len(self.weights) - 1, 0, -1):
            error = np.dot(self.weights[i].T, deltas[-1])
            delta = error * self._activate_derivative(self.layer_inputs[i], i-1)
            deltas.append(delta)
        
        deltas.reverse()
        
        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * np.outer(deltas[i], self.layer_outputs[i])
            self.biases[i] -= learning_rate * deltas[i]
    
    def train(self, X, y, epochs=1000, learning_rate=0.1, verbose=True):
        """Train the neural network"""
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            # Train on each sample
            for i in range(len(X)):
                self.backpropagate(X[i], y[i], learning_rate)
            
            # Calculate loss for monitoring
            if epoch % 10 == 0:
                predictions = np.array([self.feedforward(x) for x in X])
                loss = mse_loss(y, predictions)
                losses.append(loss)
                
                if verbose and epoch % 100 == 0:
                    print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return losses
    
    def predict(self, X):
        """Make predictions on new data"""
        return np.array([self.feedforward(x) for x in X])

# Simple Neural Network (from the tutorial)
class SimpleNeuralNetwork:
    """
    Simple 2-input, 2-hidden, 1-output neural network
    Exactly as described in the tutorial
    """
    
    def __init__(self):
        # Initialize weights randomly
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        
        # Initialize biases
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()
    
    def feedforward(self, x):
        """Forward pass through the network"""
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1
    
    def train(self, data, all_y_trues, epochs=1000, learning_rate=0.1):
        """Train the network using backpropagation"""
        losses = []
        
        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # Forward pass
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)
                
                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)
                
                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1
                
                # Calculate partial derivatives
                d_L_d_ypred = -2 * (y_true - y_pred)
                
                # Output neuron gradients
                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)
                
                d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)
                
                # Hidden neuron h1 gradients
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)
                
                # Hidden neuron h2 gradients
                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)
                
                # Update weights and biases
                self.w1 -= learning_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learning_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learning_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1
                
                self.w3 -= learning_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learning_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learning_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2
                
                self.w5 -= learning_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learning_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= learning_rate * d_L_d_ypred * d_ypred_d_b3
            
            # Record loss every 10 epochs
            if epoch % 10 == 0:
                y_preds = np.array([self.feedforward(x) for x in data])
                loss = mse_loss(all_y_trues, y_preds)
                losses.append(loss)
                
                if epoch % 100 == 0:
                    print(f"Epoch {epoch} loss: {loss:.3f}")
        
        return losses

# Example usage and testing
if __name__ == "__main__":
    print("=== Neural Network Implementation ===\n")
    
    # Example 1: Simple neuron
    print("1. Testing a single neuron:")
    weights = np.array([0, 1])
    bias = 4
    neuron = Neuron(weights, bias)
    
    x = np.array([2, 3])
    output = neuron.feedforward(x)
    print(f"Input: {x}, Output: {output:.3f}")
    
    # Example 2: Gender prediction dataset (from tutorial)
    print("\n2. Gender prediction example:")
    
    # Dataset: [weight_offset, height_offset] -> gender (0=Male, 1=Female)
    data = np.array([
        [-2, -1],   # Alice: 133-135=-2, 65-66=-1 -> Female (1)
        [25, 6],    # Bob: 160-135=25, 72-66=6 -> Male (0)
        [17, 4],    # Charlie: 152-135=17, 70-66=4 -> Male (0)
        [-15, -6],  # Diana: 120-135=-15, 60-66=-6 -> Female (1)
    ])
    
    labels = np.array([1, 0, 0, 1])  # Female=1, Male=0
    
    print("Training simple neural network...")
    simple_nn = SimpleNeuralNetwork()
    losses = simple_nn.train(data, labels, epochs=1000, learning_rate=0.1)
    
    # Test predictions
    print("\nTesting predictions:")
    emily = np.array([-7, -3])  # 128 pounds, 63 inches
    frank = np.array([20, 2])   # 155 pounds, 68 inches
    
    emily_pred = simple_nn.feedforward(emily)
    frank_pred = simple_nn.feedforward(frank)
    
    print(f"Emily: {emily_pred:.3f} ({'Female' if emily_pred > 0.5 else 'Male'})")
    print(f"Frank: {frank_pred:.3f} ({'Female' if frank_pred > 0.5 else 'Male'})")
    
    # Example 3: Flexible neural network
    print("\n3. Testing flexible neural network:")
    
    # Create a network with 2 inputs, one hidden layer with 4 neurons, 1 output
    nn = NeuralNetwork(input_size=2, hidden_sizes=[4], output_size=1, activation='sigmoid')
    
    print("Training flexible neural network...")
    losses_flexible = nn.train(data, labels, epochs=1000, learning_rate=0.1, verbose=False)
    
    # Test predictions
    predictions = nn.predict(data)
    print("Predictions on training data:")
    for i, (pred, true) in enumerate(zip(predictions, labels)):
        name = ['Alice', 'Bob', 'Charlie', 'Diana'][i]
        print(f"{name}: {pred[0]:.3f} (True: {true}, Predicted: {'Female' if pred[0] > 0.5 else 'Male'})")
    
    print("\nNew predictions:")
    emily_pred = nn.feedforward(emily)
    frank_pred = nn.feedforward(frank)
    print(f"Emily: {emily_pred:.3f} ({'Female' if emily_pred > 0.5 else 'Male'})")
    print(f"Frank: {frank_pred:.3f} ({'Female' if frank_pred > 0.5 else 'Male'})")
    
    # Plot training loss
    try:
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(range(0, len(losses) * 10, 10), losses, 'b-', label='Simple NN')
        plt.title('Simple Neural Network - Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(range(0, len(losses_flexible) * 10, 10), losses_flexible, 'r-', label='Flexible NN')
        plt.title('Flexible Neural Network - Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("\nMatplotlib not available. Skipping plot.")
        print(f"Final loss - Simple NN: {losses[-1]:.4f}")
        print(f"Final loss - Flexible NN: {losses_flexible[-1]:.4f}")