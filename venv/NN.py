import numpy as np

inputValues =[[28.67, 28.19, 21.85, 21.93, 25.47], [25.99, 20.99, 22.1, 21.02, 25.06], [23.98, 25.06, 27.32, 21.1, 18.46],
              [27.29, 26.8, 24.65, 21.41, 18.15], [32.98, 32.56, 32.42, 50.96, 77.59], [59.1, 55.7, 61.94, 77.33, 83.4],
              [32.35, 32.7, 32.27, 42.66, 79.7], [32.8, 32.85, 32.04, 44.89, 78.73], [189.01, 182.31, 177.88, 153.13, 108.36],
              [197, 186.27, 181.81, 150.02, 110.66]]
YHats = [28.61, 26.19, 24.13, 27.43, 33.08, 59.37, 32.63, 32.35, 188.93, 197.86]

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

def relu_backward(dA, Z):
    dZ = dA * np.vectorize(relu_derivative)(Z)
    return dZ;

def relu_derivative(x):
    if x >= 0:
        return 1
    else:
        return 0

nn_structure = [
    {"input": 5, "output": 7, "activation": relu},
    {"input": 7, "output": 3, "activation": relu},
    {"input": 3, "output": 3, "activation": sigmoid},
    {"input": 3, "output": 1, "activation": sigmoid}
]

nn_structure_simple = [
    {"input": 5, "output": 1, "activation": sigmoid}
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

def single_layer_forward(prevLayer, weights, biases, activation_function=sigmoid):
    currentPreAcc = activation_function(np.dot(weights, prevLayer) + biases)
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


def single_layer_backward(currLayerDiff, weights, biases, preFunc, prevLayer, activation=relu):
    m = prevLayer.shape[1]

    if activation == relu:
        a_func = relu_backward
    else:
        a_func = sigmoid_backward

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
    prevLayerDiff = -(np.divide(y, yHat) - np.divide(1-y, 1-yHat))

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
        param_changes["db" + str(currIndex)] = currBiases

    return param_changes

def update(params, param_changes, neuronStructure, learning_rate):
    for i, layer in enumerate(neuronStructure):
        layerIndex = i + 1
        params["W" + str(layerIndex)] -= param_changes["dW" + str(layerIndex)] * learning_rate
        params["b" + str(layerIndex)] -= param_changes["db" + str(layerIndex)] * learning_rate
    return params

structure = nn_structure
params = init_network(structure)
input = np.array([inputValues[1]]).T / 1000
input = np.array([inputValues[1]]).T / 1000
expectedY = np.array([[YHats[1]]]) / 1000

for i in range(10000):

    yHat, mem = forward_propagataion(input, params, structure)
    param_changes = backward_propagation(yHat, expectedY, mem, params, structure)
    params = update(params, param_changes, structure, 0.1)

    print(get_cost(yHat, expectedY))
    print(yHat)

print(params)

