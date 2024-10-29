import numpy as np

class Variable:
    def __init__(self, value):
        self.value = value          # The value of the variable
        self.grad = 0.0             # Gradient initialized to 0
        self.op = None              # Operation that generated this variable
        self.children = []          # Variables that contributed to this variable
    def __repr__(self):
        return f"Variable(value={self.value}, grad={self.grad})"

class AddOp:
    def __init__(self, left, right):
        self.left = left
        self.right = right
    def forward(self):
        result = Variable(self.left.value + self.right.value)
        result.op = self
        return result
    def backward(self, grad_output):
        self.left.grad += grad_output
        self.right.grad += grad_output

class SubOp:
    def __init__(self, left, right):
        self.left = left
        self.right = right
    def forward(self):
        result = Variable(self.left.value - self.right.value)
        result.op = self
        return result
    def backward(self, grad_output):
        self.left.grad += grad_output
        self.right.grad -= grad_output

class MulOp:
    def __init__(self, left, right):
        self.left = left
        self.right = right
    def forward(self):
        result = Variable(self.left.value * self.right.value)
        result.op = self
        return result
    def backward(self, grad_output):
        self.left.grad += grad_output * self.right.value
        self.right.grad += grad_output * self.left.value

class DivOp:
    def __init__(self, left, right):
        self.left = left
        self.right = right
    def forward(self):
        result = Variable(self.left.value / self.right.value)
        result.op = self
        return result
    def backward(self, grad_output):
        self.left.grad += grad_output / self.right.value
        self.right.grad -= grad_output * self.left.value / (self.right.value ** 2)

class PowOp:
    def __init__(self, base, exponent):
        self.base = base
        self.exponent = exponent
    def forward(self):
        result = Variable(self.base.value ** self.exponent)
        result.op = self
        return result
    def backward(self, grad_output):
        self.base.grad += grad_output * self.exponent * (self.base.value ** (self.exponent - 1))

class TanhOp:
    def __init__(self, input):
        self.input = input
        self.output = None
    def forward(self):
        self.output = Variable(np.tanh(self.input.value))       # Using numpy for the tanh function
        self.output.op = self
        return self.output

    def backward(self, grad_output):
        self.input.grad += grad_output * (1 - self.output.value ** 2)

# Integrate operations into the Variable class
class Variable:
    def __init__(self, value):
        self.value = value
        self.grad = 0.0
        self.op = None
        self.children = []
    def __repr__(self):
        return f"Variable(value={self.value}, grad={self.grad})"

    def __add__(self, other):
        op = AddOp(self, other)
        result = op.forward()
        result.children = [self, other]
        return result

    def __sub__(self, other):
        op = SubOp(self, other)
        result = op.forward()
        result.children = [self, other]
        return result

    def __mul__(self, other):
        op = MulOp(self, other)
        result = op.forward()
        result.children = [self, other]
        return result

    def __truediv__(self, other):
        op = DivOp(self, other)
        result = op.forward()
        result.children = [self, other]
        return result

    def power(self, exponent):
        op = PowOp(self, exponent)
        result = op.forward()
        result.children = [self]
        return result

    def tanh(self):
        op = TanhOp(self)
        result = op.forward()
        result.children = [self]
        return result
    
def backward(variable):
    variable.grad = 1.0  # Start with a gradient of 1 for the output
    stack = [variable]
    
    while stack:
        v = stack.pop()
        if v.op is not None:
            v.op.backward(v.grad)
            stack.extend(v.children)

# Testing

#1 Addition with Constants
x = Variable(5.0)
y = Variable(3.0)
z = x + y

backward(z)

print("Test Case 1")
print("x grad:", x.grad)        # Expected: 1
print("y grad:", y.grad)        # Expected: 1
print()

#2 Multiplication with Constants
x = Variable(4.0)
y = Variable(2.0)
z = x * y

backward(z)

print("Test Case 2")
print("x grad:", x.grad)        # Expected: 2.0 (value of y)
print("y grad:", y.grad)        # Expected: 4.0 (value of x)
print()

#3 Combination of Addition and Multiplication
x = Variable(3.0)
y = Variable(2.0)
z = x * y + y

backward(z)

print("Test Case 3")
print("x grad:", x.grad)        # Expected: 2.0 (value of y)
print("y grad:", y.grad)        # Expected: 3.0 + 1 = 4.0
print()

#4 Polynomial Function
x = Variable(3.0)
z = (x.power(2)) + (x * Variable(2.0))

backward(z)

print("Test Case 4")
print("x grad:", x.grad)        # Expected: 2 * 3.0 + 2 = 8.0
print()

#5 More Complex Composition
x = Variable(3.0)
y = Variable(1.0)
z = (x + y) * (x - y)

backward(z)

print("Test Case 5")
print("x grad:", x.grad)        # Expected: 2 * x = 6.0
print("y grad:", y.grad)        # Expected: -2 * y = -2.0
print()

#6 Power Operation
x = Variable(2.0)
n = 3
z = x.power(n)

backward(z)

print("Test Case 6")
print("Gradient of x (2^3):", x.grad)       # Expected: n * (x^(n-1)) = 3 * (2^(3-1)) = 12.0
print()

#7 tanh Operation
x = Variable(1.0)
z = x.tanh()        # Computes tanh(1)

backward(z)

print("Test Case 7")
print("Gradient of x (tanh(1)):", x.grad)       # Expected: 1 - tanh(1)^2
print()
