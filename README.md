# Deep Learning From Scratch

This repository implements a basic deep learning model from scratch using NumPy and Python. The model is trained to solve a binary classification problem using a 3-layer neural network (with one hidden layer). The code showcases the process of forward propagation, cost calculation, and backward propagation for gradient descent optimization.

## Contents

- **Data:** The model reads data from a CSV file (`Test.csv`), where the input features and labels are stored.
- **Neural Network Architecture:**
    - Input Layer: 4 features
    - Hidden Layer: 7 units
    - Output Layer: 1 unit
- **Activation Functions:** 
    - ReLU activation is applied to the hidden layer.
    - Sigmoid activation is applied to the output layer.

## Key Functions

1. **`Linear_function(W, b, X)`**: Computes the linear transformation \( A = W \cdot X + b \).
2. **`sigmoid(z)`**: Computes the sigmoid activation function.
3. **`relu(z)`**: Computes the ReLU activation function.
4. **`computed_cost(A, Y)`**: Calculates the binary cross-entropy cost between predicted values and true labels.
5. **`Linear_backward(W, dz, A)`**: Computes the gradients for backpropagation for a given layer.
6. **`sigmoid_backward(da, A)`**: Computes the backward pass for the sigmoid activation function.
7. **`Update_gradients(W, b, dW, db, learning_rate)`**: Updates the weights and biases using gradient descent.

## Workflow

1. **Data Preparation:**  
   The data is read from the provided CSV file using pandas and is split into features (X) and target labels (Y).

2. **Model Initialization:**  
   The weights and biases are randomly initialized using small values. The architecture has three layers:
   - **Layer 1 (Input to Hidden):** 4 input features to 7 hidden units
   - **Layer 2 (Hidden to Output):** 7 hidden units to 1 output unit

3. **Forward Propagation:**  
   The forward pass computes the activations for each layer using the `Linear_function` and activation functions (`sigmoid` for the output layer, `relu` for the hidden layer).

4. **Cost Calculation:**  
   The binary cross-entropy cost is computed at each iteration to evaluate model performance.

5. **Backward Propagation:**  
   Backpropagation is used to compute the gradients for weights and biases using chain rule and updates the parameters using gradient descent.

6. **Training Loop:**  
   The model runs for 50 iterations, updating the weights and biases after each iteration to minimize the cost.

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/Deep-Learning-From-Scratch.git
