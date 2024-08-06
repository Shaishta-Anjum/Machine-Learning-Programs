import numpy as np


# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)


# Initialize the neural network
def initialize_network(input_size, hidden_size, output_size):
    np.random.seed(1)
    weights_input_hidden = np.random.rand(input_size, hidden_size)
    weights_hidden_output = np.random.rand(hidden_size, output_size)
    return weights_input_hidden, weights_hidden_output


# Forward propagation
def forward_propagate(inputs, weights_input_hidden, weights_hidden_output):
    hidden_input = np.dot(inputs, weights_input_hidden)
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, weights_hidden_output)
    final_output = sigmoid(final_input)
    return hidden_output, final_output


# Backpropagation
def backpropagate(inputs, hidden_output, final_output, expected_output, weights_input_hidden, weights_hidden_output, learning_rate):
    output_error = expected_output - final_output
    output_delta = output_error * sigmoid_derivative(final_output)

    hidden_error = output_delta.dot(weights_hidden_output.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_output)

    weights_hidden_output += hidden_output.T.dot(output_delta) * learning_rate
    weights_input_hidden += inputs.T.dot(hidden_delta) * learning_rate

    return weights_input_hidden, weights_hidden_output


# Training the network
def train_network(inputs, expected_output, weights_input_hidden, weights_hidden_output, learning_rate, epochs):
    for epoch in range(epochs):
        hidden_output, final_output = forward_propagate(inputs, weights_input_hidden, weights_hidden_output)
        weights_input_hidden, weights_hidden_output = backpropagate(inputs, hidden_output, final_output, expected_output, weights_input_hidden, weights_hidden_output, learning_rate)
        if epoch % 1000 == 0:
            loss = np.mean(np.square(expected_output - final_output))
            print(f"Epoch {epoch}, Loss: {loss}")
    return weights_input_hidden, weights_hidden_output


# Test the network
def test_network(inputs, weights_input_hidden, weights_hidden_output):
    _, final_output = forward_propagate(inputs, weights_input_hidden, weights_hidden_output)
    return final_output


# Sample data (XOR problem)
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
expected_output = np.array([[0], [1], [1], [0]])

# Initialize network
input_size = 2
hidden_size = 2
output_size = 1
weights_input_hidden, weights_hidden_output = initialize_network(input_size, hidden_size, output_size)

# Train network
learning_rate = 0.1
epochs = 10000
weights_input_hidden, weights_hidden_output = train_network(inputs, expected_output, weights_input_hidden, weights_hidden_output, learning_rate, epochs)

# Test network
final_output = test_network(inputs, weights_input_hidden, weights_hidden_output)
print("Final Output:")
print(final_output)