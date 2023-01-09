#include "../inc/declare.h"

// Simple network that can learn XOR
// Feartures : sigmoid activation function, stochastic gradient descent, and mean square error fuction

// Activation function and its derivative
double sigmoid(double x) { return 1 / (1 + exp(-x)); }
double dSigmoid(double x) { return x * (1 - x); }
// Activation function and its derivative
double init_weight() { return ((double)rand()) / ((double)RAND_MAX); }

int main(void) {

    const double learningrate = 0.1f;

    int numHiddenNodes=0;
    printf("Enter the number of HiddenNodes:");
    scanf("%d", &numHiddenNodes);

    double* hiddenLayer = (double*)malloc(numHiddenNodes*sizeof(double));
    double outputLayer[numOutputs];

    double* hiddenLayerBias =(double*)malloc(numHiddenNodes * sizeof(double));
    double outputLayerBias[numOutputs];

    double* hiddenWeights = (double*)malloc( numInputs * numHiddenNodes * sizeof(double));
    double* outputWeights = (double*)malloc(numHiddenNodes * numOutputs* sizeof(double));


    double training_inputs[numTrainingSets][numInputs] = { {0.0f,0.0f},
                                                          {1.0f,0.0f},
                                                          {0.0f,1.0f},
                                                          {1.0f,1.0f} };
    double training_outputs[numTrainingSets][numOutputs] = { {0.0f},
                                                            {1.0f},
                                                            {1.0f},
                                                            {0.0f} };
    double loss;
    double losssquare=0;
    //store the initial weight in the dynamic memory
    for (int i = 0; i < numInputs; i++) {
        for (int j = 0; j < numHiddenNodes; j++) {
            int index = i * numHiddenNodes + j;
            hiddenWeights[index] = init_weight();
        }
    }
    //store the hidden layer bias in the dynamic memory
    //store the output  weight in the dynamic memory
    for (int i = 0; i < numHiddenNodes; i++) {
        hiddenLayerBias[i] = init_weight();
        for (int j = 0; j < numOutputs; j++) {
            int index = i * numOutputs + j;
            outputWeights[index] = init_weight();
        }
    }
    //store the output layer bias in a matrix 
    for (int i = 0; i < numOutputs; i++) {
        outputLayerBias[i] = init_weight();
    }

    int trainingSetOrder[] = { 0,1,2,3 };

    int numberOfEpochs = 30000;
    // Train the neural network for a number of epochs
    for (int epochs = 0; epochs < numberOfEpochs; epochs++) {

        // As per SGD, shuffle the order of the training set
        shuffle(trainingSetOrder, numTrainingSets);

        // Cycle through each of the training set elements
        for (int x = 0; x < numTrainingSets; x++) {

            int i = trainingSetOrder[x];

            // Forward pass

            // Compute hidden layer activation
            for (int j = 0; j < numHiddenNodes; j++) {
                double activation = hiddenLayerBias[j];
                for (int k = 0; k < numInputs; k++) {
                    int index = j * numInputs + k;
                    activation += training_inputs[i][k] * hiddenWeights[index];
                }
                hiddenLayer[j] = sigmoid(activation);
            }

            // Compute output layer activation
            for (int j = 0; j < numOutputs; j++) {
                double activation = outputLayerBias[j];
                for (int k = 0; k < numHiddenNodes; k++) {
                    int index = j * numHiddenNodes + k;
                    activation += hiddenLayer[k] * outputWeights[index];
                }
                outputLayer[j] = sigmoid(activation);
                loss = (outputLayer[j]-training_outputs[i][j]);
            }
            losssquare = losssquare+pow(loss,2);

            // Print the results from forward pass
            printf("Input:%g %g  Output:%g   Expected Output: %g,  loss:%g\n",
              training_inputs[i][0], training_inputs[i][1],
              outputLayer[0], training_outputs[i][0],loss);
           
             
            // Backprop

            // Compute change in output weights
            double deltaOutput[numOutputs];
            for (int j = 0; j < numOutputs; j++) {
                double errorOutput = (training_outputs[i][j] - outputLayer[j]);
                deltaOutput[j] = errorOutput * dSigmoid(outputLayer[j]);
            }

            // Compute change in hidden weights
            double* deltaHidden = (double*)malloc(numHiddenNodes * sizeof(double));
            for (int j = 0; j < numHiddenNodes; j++) {
                double errorHidden = 0.0f;
                for (int k = 0; k < numOutputs; k++) {
                    int index = j * numOutputs + k;
                    errorHidden += deltaOutput[k] * outputWeights[index];
                }
                deltaHidden[j] = errorHidden * dSigmoid(hiddenLayer[j]);
            }

            // Apply change in output weights
            for (int j = 0; j < numOutputs; j++) {
                outputLayerBias[j] += deltaOutput[j] * learningrate;
                for (int k = 0; k < numHiddenNodes; k++) {
                    int index = j * numHiddenNodes + k;
                    outputWeights[index] += hiddenLayer[k] * deltaOutput[j] * learningrate;
                }
            }

            // Apply change in hidden weights
            for (int j = 0; j < numHiddenNodes; j++) {
                hiddenLayerBias[j] += deltaHidden[j] * learningrate;
                for (int k = 0; k < numInputs; k++) {
                    int index = j * numInputs + k;
                    hiddenWeights[index] += training_inputs[i][k] * deltaHidden[j] * learningrate;
                }
            }
        }
    }

    // Print final weights after training
    fputs("Final Hidden Weights\n[ ", stdout);
    for (int j = 0; j < numHiddenNodes; j++) {
        fputs("[ ", stdout);
        for (int k = 0; k < numInputs; k++) {
            int index = j * numInputs + k;
            printf("%f ", hiddenWeights[index]);
        }
        fputs("] ", stdout);
    }

    fputs("]\nFinal Hidden Biases\n[ ", stdout);
    for (int j = 0; j < numHiddenNodes; j++) {
        printf("%f ", hiddenLayerBias[j]);
    }

    fputs("]\nFinal Output Weights", stdout);
    for (int j = 0; j < numOutputs; j++) {
        fputs("[ ", stdout);
        for (int k = 0; k < numHiddenNodes; k++) {
            int index = j * numHiddenNodes + k;
            printf("%f ", outputWeights[index]);
        }
        fputs("]\n", stdout);
    }

    fputs("Final Output Biases\n[ ", stdout);
    for (int j = 0; j < numOutputs; j++) {
        printf("%f ", outputLayerBias[j]);

    }
    
    fputs("]\n", stdout);
    printf("loss function:%f\n\n", losssquare/ numberOfEpochs);
    return 0;
    free(hiddenLayer);
    free(hiddenLayerBias);
    free(outputWeights);
    free(hiddenWeights);
}