using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class Network
    {
        public List<Layer> layers = new List<Layer>();

        public Network(List<int> numNodes)
        {
            bool isInput = true; //defines the first layer as the input layer

            int layerCount = 0; //keeps track of which layer is currently being created

            int numNodesInPreLayer = 0;

            foreach (int numNodesInCurLayer in numNodes)
            {
                layerCount++;

                if (isInput)    //if this is the first layer
                {
                    layers.Add(new Layer(numNodesInCurLayer)); //add an input layer to layers list
                    isInput = false;    //sets isInput to false so no more input layers can be created
                }
                else if (layerCount == numNodes.Count)   //if this is the last layer an output layer must be created
                {
                    layers.Add(new Layer(numNodesInCurLayer, numNodesInPreLayer, true));    //the defineOuput is true because this is the output layer
                }
                else    //the layer must be a hidden layer
                {
                    layers.Add(new Layer(numNodesInCurLayer, numNodesInPreLayer, false));   //the defineOutput is false because this is a hidden layer
                }

                numNodesInPreLayer = numNodesInCurLayer;    //this sets the number of nodes in the previous layer to the correct value for the next iteration
            }
        }

        /// <summary>
        /// trains the network with a list of input and outputs, and allows
        /// for changing of the learning rate and tests per update to observe changes in learning
        /// </summary>
        /// <param name="input"></param>
        /// <param name="output"></param>
        /// <param name="learningRate"></param>
        /// <param name="testsPerUpdate"></param>
        public void train(List<double[]> input, List<double[]> output, double learningRate, int testsPerUpdate)
        {
            for (int i = 0; i < (input.Count / testsPerUpdate); i++) //loop for each update
            {
                for (int j = 0; j < testsPerUpdate; j++) //loop to calculate derivatives for each set of data
                {
                    calcDerivatives(input[(i * testsPerUpdate + j)], output[(i * testsPerUpdate + j)]); //calcs derivatives for set of data
                }
                updateWeightsAndBiases(learningRate, testsPerUpdate); //updates the weights and biases
            }
        }

        /// <summary>
        /// calculates all derivatives for the network
        /// first the network values are determined with the specific input
        /// then the neuron derivatives are calculated first
        /// the the weights derivatives and bias derivatives can be calculated
        /// the neuron derivative must be first because it is required in the bias and weight derivatives
        /// </summary>
        /// <param name="input"></param>
        /// <param name="output"></param>
        public void calcDerivatives(double[] input, double[] output)
        {
            calcNetwork(input);
            calcActivDerivZ();
            calcNeuronDeriv(output);
            calcWeightDeriv();
            calcBiasDeriv();
        }

        public void calcActivDerivZ()
        {
            for (int i = 1; i < layers.Count; i++)
            {
                layers[i].activDerivZ = NetworkMath.activationFunctionDeriv(layers[i].z);
            }
        }

        public void updateWeightsAndBiases(double rate, double numTests)
        {
            for (int layerNum = 1; layerNum < layers.Count; layerNum++)    //loops through each layers starting with the first hidden layer
            {   //WEIGHT UPDATE
                double[,] sumWeight = new double[layers[layerNum].derivWeight[0].GetLength(0), layers[layerNum].derivWeight[0].GetLength(1)]; //holds sum for the weight derivatives

                for (int i = 0; i < sumWeight.GetLength(0); i++)   //this goes through the first index
                {
                    for (int j = 0; j < sumWeight.GetLength(1); j++)   //this goes throguh the second index
                    {
                        sumWeight[i, j] = 0;  //sets every value to zero
                    }
                }

                foreach (double[,] d in layers[layerNum].derivWeight)   //loops through each derivative in the list and adds it to the sum
                {
                    sumWeight = NetworkMath.add(sumWeight, d);  //adds the derivatives to the sum
                }

                for (int i = 0; i < sumWeight.GetLength(0); i++)   //this goes through the first index
                {
                    for (int j = 0; j < sumWeight.GetLength(1); j++)   //this goes throguh the second index
                    {
                        sumWeight[i, j] /= numTests; //divides the sum by number of tests to calc average

                        layers[layerNum].weight[i, j] -= sumWeight[i, j] * rate;  //the actual update of each weight
                    }
                }

                //BIAS UPDATE
                double[] sumBias = new double[layers[layerNum].derivBias[0].GetLength(0)];  //creates a sum array for the bias derivatives

                for (int i = 0; i < sumBias.Length; i++)     //loops through the bias sum array
                {
                    sumBias[i] = 0; //sets every element in the array to zero
                }

                foreach (double[] d in layers[layerNum].derivBias)   //loops through all the derivatives from every test
                {
                    sumBias = NetworkMath.add(sumBias, d);  //adds the derivatives to the sum
                }

                for (int i = 0; i < sumBias.Length; i++) //loops through each element of the sum array
                {
                    sumBias[i] /= numTests; //divides the sum by the number of tests to calculate the average

                    layers[layerNum].bias[i] -= sumBias[i] * rate;  //the actual update of each bias
                }


                layers[layerNum].derivWeight.Clear(); //removes the list of derivatives used in this update
                layers[layerNum].derivBias.Clear();
                layers[layerNum].derivNeuron.Clear();
            }
        }

        public void calcNeuronDeriv(double[] output)
        {
            List<double> tempNeuronDeriv = new List<double>();
            double tempSum;

            for (int i = 0; i < layers[layers.Count - 1].neuron.Length; i++)  //loop through all nuerons in the last layer
            {
                tempNeuronDeriv.Add(layers[layers.Count - 1].neuron[i] - output[i]);    //adds each derivative to the temporoary list
            }

            layers[layers.Count - 1].derivNeuron.Add(tempNeuronDeriv.ToArray());    //adds the list of temporary 

            for (int layerNum = layers.Count - 2; layerNum > 0; layerNum--)  //loop through each layer backwards 'BACK propogation'
            {
                tempNeuronDeriv.Clear();    //clear the temporary list of derivatives before calculating all the neurons

                for (int i = 0; i < layers[layerNum].neuron.Length; i++) //loop through each neuron in a layer to calc its specific derivative
                {
                    tempSum = 0.0;
                    for (int j = 0; j < layers[layerNum + 1].neuron.Length; j++)   //calculate the sumation of weight*activation'(z)*previousNueron' (equation for derivative of neurons beyond last layer)
                    {
                        tempSum += layers[layerNum + 1].weight[j, i] *
                            layers[layerNum + 1].activDerivZ[j] *
                            layers[layerNum + 1].derivNeuron[layers[layerNum + 1].derivNeuron.Count - 1][j];
                    }
                    tempNeuronDeriv.Add(tempSum);   //add to the list of neuron derivs for this layer
                }

                layers[layerNum].derivNeuron.Add(tempNeuronDeriv.ToArray());    //sets the current layer's neuron derivative = to its corresponding value after calculating all sumations
            }
        }

        public void calcWeightDeriv()
        {
            for (int layerNum = layers.Count - 1; layerNum > 0; layerNum--)    //loop through each layer backwards 'Back propogation'
            {
                double[,] tempDeriv = new double[layers[layerNum].weight.GetLength(0), layers[layerNum].weight.GetLength(1)];   //create temporary 2d array for the derivatives

                for (int curIndex = 0; curIndex < layers[layerNum].weight.GetLength(0); curIndex++)    //loops through all the weights from their current layer index
                {
                    for (int preIndex = 0; preIndex < layers[layerNum].weight.GetLength(1); preIndex++) //loops through all the weights from their previous layer index
                    {
                        tempDeriv[curIndex, preIndex] = (layers[layerNum - 1].neuron[preIndex] *
                            layers[layerNum].activDerivZ[curIndex] *
                            layers[layerNum].derivNeuron[layers[layerNum].derivNeuron.Count - 1][curIndex]);    //does the calculation for a specific neuron
                    }
                }

                layers[layerNum].derivWeight.Add(tempDeriv);    //adds deriv array to current layers list of deriv arrays
            }
        }

        public void calcBiasDeriv()
        {
            for (int layerNum = layers.Count - 1; layerNum > 0; layerNum--)  //loop through each layer backwards 'Back propogation'
            {
                double[] tempDeriv = new double[layers[layerNum].bias.Length];  //create temporary list for the derivatives of the current layer

                for (int i = 0; i < layers[layerNum].bias.Length; i++)   //loop through all indices of current set of biases
                {
                    tempDeriv[i] = layers[layerNum].activDerivZ[i] *
                        layers[layerNum].derivNeuron[layers[layerNum].derivNeuron.Count - 1][i];    //calculation for bias derivative
                }

                layers[layerNum].derivBias.Add(tempDeriv);  //adds deriv array to current layers list of deriv arrays
            }
        }


        /// <summary>
        /// calculates the value for all neurons in the network
        /// then returns the values of the neurons in the last layer (ie the output layer)
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public double[] calcOutput(double[] input)
        {
            calcNetwork(input);

            return layers[layers.Count - 1].neuron;   // returns
        }

        /// <summary>
        /// calculates all neuron values in the network
        /// </summary>
        /// <param name="input"></param>
        public void calcNetwork(double[] input)
        {
            layers[0].neuron = input;   //sets the correct values of the nodes of the input layer

            for (int i = 1; i < (layers.Count); i++)   //loops through each layer (starting with the first hidden layer)
            {
                //calcLayer(layers[i - 1], layers[i]);
                double[] temp = NetworkMath.multiply(layers[i].weight, layers[i - 1].neuron); //multiplies the weights and the value of the previous layers neurons

                layers[i].z = NetworkMath.add(layers[i].bias, temp);   //adds the appropriate biases to the product of the weights and neurons

                layers[i].neuron = NetworkMath.activationFunction(layers[i].z);    //applies an activation function on the previous sum            
            }
        }

        /// <summary>
        /// calculates the value of a single layer's neuron values
        /// </summary>
        /// <param name="preLayer"></param>
        /// <param name="curLayer"></param>
        public void calcLayer(Layer preLayer, Layer curLayer)
        {
            double[] temp = NetworkMath.multiply(curLayer.weight, preLayer.neuron); //multiplies the weights and the value of the previous layers neurons

            curLayer.z = NetworkMath.add(curLayer.bias, temp);   //adds the appropriate biases to the product of the weights and neurons

            curLayer.neuron = NetworkMath.activationFunction(curLayer.z);    //applies an activation function on the previous sum            
        }

        /// <summary>
        /// displays the following values for all layers on the console:
        ///     weights
        ///     biases
        ///     neuron values
        /// </summary>
        public void rawOutput()
        {
            int layerNum = 0;

            Console.WriteLine("LAYER: INPUT");
            Console.WriteLine("VALUES");
            foreach (double v in layers[0].neuron)
            {
                Console.WriteLine(v);
            }

            for (int index = 1; index < (layers.Count); index++)   //loops through each layer (starting with the first hidden layer)
            {
                Layer layer = layers[index];

                layerNum++;
                Console.WriteLine("LAYER: " + layerNum);
                Console.WriteLine("WEIGHTS");
                for (int i = 0; i < (layer.weight.GetLength(0)); i++)    //loop through the weights and display each nodes list of weights in a row
                {
                    for (int j = 0; j < (layer.weight.GetLength(1)); j++)
                    {
                        Console.Write(layer.weight[i, j] + ", ");
                    }
                    Console.WriteLine("");
                }
                Console.WriteLine("");

                Console.WriteLine("BIAS");
                for (int i = 0; i < (layer.bias.Length); i++)    //loop through all biases for the current layer
                {
                    Console.Write(layer.bias[i] + ", ");
                }
                Console.WriteLine("");
                Console.WriteLine("");

                Console.WriteLine("CURRENT NEURON VALUES");
                for (int i = 0; i < (layer.neuron.Length); i++) //loop through all neuron values for the current layer (calculated at last calc neuron
                {
                    Console.Write(layer.neuron[i] + ", ");
                }
                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine("");
            }
        }
    }
}
