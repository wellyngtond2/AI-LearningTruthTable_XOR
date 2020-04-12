import numpy as np

_inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
_outputs = np.array([[0],[1],[1],[0]])

_weight0 = np.array([[-0.424,-0.740,-0.961],
                    [0.358,-0.577,-0.469]])

_weight1 = np.array([[-0.017],[-0.893],[0.148]])

numberTraining =1000000
learningRate= 0.3
momentum=1

def Sigmoide(sum):
    return 1 / (1 + np.exp(-sum))

def SigmoideDerivate(sig):
    return sig * (1-sig)

sigDerivate = Sigmoide(0.5)
sigDerivate1 = SigmoideDerivate(sigDerivate)


for i in range(numberTraining):
    inputLayer = _inputs
    sinapseSum0 = np.dot(inputLayer,_weight0)
    hiddenLayer = Sigmoide(sinapseSum0)
    
    sinapseSum1 = np.dot(hiddenLayer,_weight1)
    outputLayer = Sigmoide(sinapseSum1)
    
    erroOutputLayer = _outputs - outputLayer
    absoluteAverage = np.mean(np.abs(erroOutputLayer))       

    outputDerivate = SigmoideDerivate(outputLayer)
    outputDelta = erroOutputLayer * outputDerivate    
    
    weight1transposed = _weight1.T
    deltaOutputXweights = outputDelta.dot(weight1transposed)
    deltaHiddenLayer = deltaOutputXweights * SigmoideDerivate(hiddenLayer)
    
    hiddenLayerTransposed = hiddenLayer.T
    weight3 = hiddenLayerTransposed.dot(outputDelta)    
    _weight1 = (_weight1 * momentum) + (weight3 * learningRate)
    
    inputLayerTransposed = inputLayer.T
    weight4 = inputLayerTransposed.dot(deltaHiddenLayer)
    _weight0 = (_weight0 * momentum) + (weight4 * learningRate)
    
    print('Absolute avarage:' + str(absoluteAverage))