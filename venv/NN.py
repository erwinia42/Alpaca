import numpy as np
import math, json
from tqdm import tqdm
from TestingSeq import get_training_data, get_test_train_data

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.vectorize(math.tanh)(x)

def lin(x):
    return x

def x3(x):
    return x**3

def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

def relu_backward(dA, Z):
    dZ = dA * np.vectorize(relu_derivative)(Z)
    return dZ

def tanh_backward(dA, Z):
    tan = tanh(Z)
    return dA * (1 - tan**2)

def lin_backward(dA, Z):
    return dA

def x3_backward(dA, Z):
    return dA * 3 * (Z**2)

def relu_derivative(x):
    if x >= 0:
        return 1
    else:
        return 0

nn_structure = [
    {"input": 8, "output": 128, "activation": relu},
    {"input": 128, "output": 24, "activation": tanh},
    {"input": 24, "output": 5, "activation": tanh},
    {"input": 5, "output": 1, "activation": tanh}
]

nn_structure_simple = [
    {"input": 6, "output": 1, "activation": tanh}
]

def init_network(neuronStructure):
    layers = len(neuronStructure)
    params = {}

    for i, layer in enumerate(neuronStructure):
        layerIndex = i + 1
        input_size = layer["input"]
        output_size = layer["output"]

        params["W" + str(layerIndex)] = np.random.randn(
            output_size, input_size) * 0.1
        params["b" + str(layerIndex)] = np.random.randn(
            output_size, 1) * 0.1

    return params

def single_layer_forward(prevLayer, weights, biases, activation_function=tanh):
    currentPreAcc = np.dot(weights, prevLayer) + biases
    return activation_function(currentPreAcc), currentPreAcc

def forward_propagataion(input, params, neuronStructure):
    mem = {}
    currLayer = input

    for i, layer in enumerate(neuronStructure):
        layerIndex = i + 1
        prevLayer = currLayer

        weights = params["W" + str(layerIndex)]
        biases = params["b" + str(layerIndex)]
        a_func = neuronStructure[i]["activation"]
        currLayer, currPreFunc = single_layer_forward(prevLayer, weights, biases, activation_function=a_func)

        mem["A" + str(i)] = prevLayer
        mem["Z" + str(layerIndex)] = currPreFunc

    return currLayer, mem

def get_cost(yHat, y):
    return (yHat-y)**2


def single_layer_backward(currLayerDiff, weights, biases, preFunc, prevLayer, activation=tanh):
    m = prevLayer.shape[1]

    if activation == relu:
        a_func = relu_backward
    elif activation == tanh:
        a_func = tanh_backward
    elif activation == lin:
        a_func = lin_backward
    elif activation == sigmoid:
        a_func = sigmoid_backward
    elif activation == x3:
        a_func = x3_backward

    #Math is hard
    preFuncDiff = a_func(currLayerDiff, preFunc)
    weightDiff = np.dot(preFuncDiff, prevLayer.T) / m
    biasDiff = np.sum(preFuncDiff, axis=1, keepdims=True) / m
    prevLayerDiff = np.dot(weights.T, preFuncDiff)

    return prevLayerDiff, weightDiff, biasDiff

def backward_propagation(yHat, y, mem, params, neuronStructure):
    param_changes = {}
    y = y.reshape(yHat.shape)

    #The first diff
    prevLayerDiff = 2*(yHat - y)

    for prevIndex, layer in reversed(list(enumerate(neuronStructure))):
        currIndex = prevIndex + 1
        a_func = layer["activation"]

        currLayerDiff = prevLayerDiff

        prevLayer = mem["A" + str(prevIndex)]
        currPreFunc = mem["Z" + str(currIndex)]
        currWeights = params["W" + str(currIndex)]
        currBiases = params["b" + str(currIndex)]

        prevLayerDiff, currWeightDiff, currBiasDiff = single_layer_backward(currLayerDiff, currWeights, currBiases,
                                                                        currPreFunc, prevLayer, activation=a_func)

        param_changes["dW" + str(currIndex)] = currWeightDiff
        param_changes["db" + str(currIndex)] = currBiasDiff

    return param_changes

def update(params, param_changes, neuronStructure, learning_rate):
    for i, layer in enumerate(neuronStructure):
        layerIndex = i + 1
        params["W" + str(layerIndex)] -= param_changes["dW" + str(layerIndex)] * learning_rate
        params["b" + str(layerIndex)] -= param_changes["db" + str(layerIndex)] * learning_rate
    return params

def train_network(ys, xs, structure, learning_rate, epochs):
    params = init_network(structure)
    cost_history = []

    for j in range(epochs):
        for i in tqdm(range(len(xs)), desc="Training Network..." + str(j + 1) + "/" + str(epochs)):
            input = np.array([xs[i]]).T
            y = np.array([ys[i]])

            yHat, mem = forward_propagataion(input, params, structure)

            param_changes = backward_propagation(yHat, y, mem, params, structure)
            params = update(params, param_changes, structure, learning_rate)

            cost_history.append(get_cost(yHat, y))

    return params, cost_history

def test_network(testys, testxs, params, structure):
    costs = []
    best = []
    worst = []
    values = {}
    for i in range(10):
        values[str(i/10)] = []
    for i in range(len(testxs)):
        input = np.array([testxs[i]]).T
        y = np.array([testys[i]])

        yHat, _ = forward_propagataion(input, params, structure)
        costs.append(get_cost(yHat, y))

        for i in range(10):
            if yHat > (i / 10):
                values[str(i/10)].append((yHat, y))

        if len(best) < 100:
            best.append((yHat, y))
        elif (yHat > min(best)).any():
            best.remove(min(best))
            best.append((yHat, y))

        if len(worst) < 100:
            worst.append((yHat, y))
        elif(yHat < max(worst)).any():
            worst.remove(max(worst))
            worst.append((yHat, y))

    for i in range(10):
        val = 0
        guesses = values[str(i/10)]
        for guess, correct in guesses:
            val += correct / len(guesses)
        print(i/10, val)

    return costs, best, worst

def ask_network(x, params, structure):
    input = np.array([x]).T
    guess, _ = forward_propagataion(input, params, structure)
    return guess

def get_params():
    paramArray = {}

    with open("params.json", "r") as f:
        paramJSON = json.load(f)

    for key, value in paramJSON.items():
        paramArray[key] = np.array(value)
    return paramArray

def write_params(params):
    paramJSON = {}
    for key, value in params.items():
        paramJSON[key] = value.tolist()

    with open("params.json", "w") as f:
        json.dump(paramJSON, f, sort_keys=False, indent=4)

def main():
    ys, xs, testys, testxs = get_test_train_data(1)
    #params, cost_history = train_network(ys, xs, nn_structure, 0.001, 10)
    #print("Training Start:", sum(cost_history[:1000]) / len(cost_history[:1000]))
    #print("Training End:", sum(cost_history[-1000:]) / len(cost_history[-1000:]))

    #write_params(params)
    params = get_params()
    test_cost, bestVals, worstVals = test_network(testys, testxs, params, nn_structure)
    print("Test:", sum(test_cost) / len(test_cost))

    res = 0
    for guess, correct in bestVals:
        print(guess, correct)
        res += correct / len(bestVals)
    print("Res+:", res)

    res = 0
    for guess, correct in worstVals:
        res += correct / len(worstVals)
    print("Res-:", res)

if __name__ == '__main__':
        main()