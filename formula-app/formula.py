from flask import Flask, render_template
import torch
import os

app = Flask(__name__)

def crossEntropyLoss(output, target):
    exp_output = torch.exp(output - torch.max(output))
    probabilities = exp_output / torch.sum(exp_output)
    log_probabilities = torch.log(probabilities + 1e-10)
    return -torch.sum(target * log_probabilities)

def meanSquareError(output, target):
    return torch.mean((output - target) ** 2)

def binaryEntropyLoss(output, target):
    loss = - (target * torch.log(output + 1e-10) + (1 - target) * torch.log(1 - output + 1e-10))
    return torch.mean(loss)

def sigmoid(x: torch.Tensor):
    return 1 / (1 + torch.exp(-x))

def relu(x: torch.tensor):
    return torch.maximum(x, torch.tensor(0.0))

def softmax(zi: torch.tensor):
    exp_zi = torch.exp(zi - torch.max(zi))
    return exp_zi / torch.sum(exp_zi)

def tanh(x: torch.tensor):
    return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))

@app.route('/')
def index():
    inputs = torch.tensor([0.1, 0.3, 0.6, 0.7])
    target = torch.tensor([0.31, 0.32, 0.8, 0.2])

    mse = meanSquareError(inputs, target).item()
    binary_loss = binaryEntropyLoss(inputs, target).item()
    cross_loss = crossEntropyLoss(inputs, target).item()

    x = torch.tensor([1, 5, -4, 3, -2])
    sigmoid_vals = sigmoid(x).tolist()
    relu_vals = relu(x).tolist()
    softmax_vals = softmax(x).tolist()
    tanh_vals = tanh(x).tolist()

    return render_template('form.html',
                           mse=mse,
                           binary_loss=binary_loss,
                           cross_loss=cross_loss,
                           f_sigmoid=sigmoid_vals,
                           f_relu=relu_vals,
                           f_softmax=softmax_vals,
                           f_tanh=tanh_vals)

if __name__ == '__main__':
    app.run(debug=True)
