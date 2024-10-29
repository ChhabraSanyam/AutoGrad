# A Simple AutoGrad System
Implementing a simple automatic differentiation system similar to Andrej Karpathy's Micrograd. This system will track operations in a computational graph and compute gradients automatically for various mathematical operations. 


## Overview

This autograd system implements automatic differentiation to compute gradients for mathematical operations involving `Variable` objects. It builds a computational graph during the forward pass, which is then traversed in the backward pass to calculate gradients using the chain rule. This system is foundational for training machine learning models, as it simplifies the computation of gradients necessary for optimizing model parameters.

## Key Components

### 1. Variable Class

The `Variable` class represents a scalar value and its gradient. Each variable can store the result of a mathematical operation and track the operation that generated it.

#### Attributes:
- **value**: The numeric value of the variable.
- **grad**: The gradient of the variable (initialized to `0.0`).
- **op**: The operation that generated this variable (if any).
- **children**: A list of child variables (inputs) used in the operation.

#### Methods:
- `__add__(self, other)`: Implements the addition operation.
- `__sub__(self, other)`: Implements the subtraction operation.
- `__mul__(self, other)`: Implements the multiplication operation.
- `__truediv__(self, other)`: Implements the division operation.
- `power(self, exponent)`: Implements the power operation.
- `tanh(self)`: Implements the tanh activation function.

### 2. Operation Classes

Each mathematical operation is represented by a dedicated class that encapsulates the logic for the operation itself and how to compute gradients.

#### a. AddOp Class
Represents the addition operation.
- **Methods**:
  - `forward()`: Computes the sum of two variables.
  - `backward(grad_output)`: Computes gradients with respect to the input variables.

#### b. SubOp Class
Represents the subtraction operation.
- **Methods**:
  - `forward()`: Computes the difference between two variables.
  - `backward(grad_output)`: Computes gradients for subtraction.

#### c. MulOp Class
Represents the multiplication operation.
- **Methods**:
  - `forward()`: Computes the product of two variables.
  - `backward(grad_output)`: Computes gradients for multiplication.

#### d. DivOp Class
Represents the division operation.
- **Methods**:
  - `forward()`: Computes the quotient of two variables.
  - `backward(grad_output)`: Computes gradients for division.

#### e. PowOp Class
Represents the power operation.
- **Methods**:
  - `forward()`: Computes the power of a variable.
  - `backward(grad_output)`: Computes gradients for power.

#### f. TanhOp Class
Represents the tanh activation function.
- **Methods**:
  - `forward()`: Computes the tanh of a variable.
  - `backward(grad_output)`: Computes gradients for tanh.

## Functionality

### Forward Pass

The forward pass involves evaluating a series of operations to compute a final output. During this process, each operation creates a new `Variable` that stores the result and the operation itself, building a computational graph.

### Backward Pass

The backward pass calculates the gradients of each variable involved in the computation. This is done by traversing the computational graph in reverse and applying the chain rule:

1. **Starting Point**: Begin with the output variable, initializing its gradient to `1.0`.
2. **Traversal**: For each operation in the graph:
   - Compute the gradient with respect to its inputs using the `backward()` method of the operation class.
   - Update the gradients in the `grad` attribute of each `Variable` that contributed to the operation.

## Usage

Various test cases have been used to validate the correctness of the autograd system:

	•	Test Case 1: Basic addition and gradient checks.
	•	Test Case 2: Basic multiplication and gradient checks.
	•	Test Case 3: Combination of addition and multiplication.
	•	Test Case 4: Polynomial functions.
	•	Test Case 5: More complex expressions involving both addition and subtraction.
	•	Test Case 6: Power operation.
	•	Test Case 7: Tanh operation.


