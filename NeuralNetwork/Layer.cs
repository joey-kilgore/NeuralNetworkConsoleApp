using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class Layer
    {
        public double[,] weight;  //weight matrix
                                  // weight[r][c]
                                  // r - neuron index current layer
                                  // c - neuron index previous layer

        public double[] neuron;    //stores values of the neurons in the layer

        public double[] bias;  //stores values of bias for each neuron in the layer

        public double[] z;  //stores the values used in the activation function
                            //these values will also be needed in the calcuations during back propogation

        public double[] activDerivZ;    //stores the values used in back propogation
                                        //calculating values once improves speed from calculating it multiple times for 
                                        //neuron, bias, and weight derivatives

        bool isInput;   //is true if the layer is an input layer
        bool isOuput;   //is true if the layer is an output layer

        public List<double[,]> derivWeight = new List<double[,]>();    //stores a list of 2d arrays corresponding to
                                                                       //the derivatives of the respective weights of 
                                                                       //the cost function

        public List<double[]> derivBias = new List<double[]>();    //stores a list of 1d arrays corresponding to the derivatives
                                                                   //of the respective biases to the cost function

        public List<double[]> derivNeuron = new List<double[]>();  //stores a list of 1d arrays corresponding to the derivatives
                                                                   //of the respective neurons to the cost function

        public Layer(int inputNumNodes)
        {
            isInput = true; //this is an input layer
            isOuput = false;

            neuron = new double[inputNumNodes]; //sets the appropriate size of the neuron array
        }

        public Layer(int currentNumNodes, int previousNumNodes, bool defineOuput)
        {
            isInput = false;    //this is a output layer if defineOutput is set to true
            isOuput = defineOuput;

            weight = new double[currentNumNodes, previousNumNodes];  //sets the appropriate size of the weight array

            neuron = new double[currentNumNodes];   //sets the appropriate size of the neuron array

            bias = new double[currentNumNodes]; //sets the appropriate size of the bias array

            setRandomWeightsAndBiases(this);
        }

        public void setRandomWeightsAndBiases(Layer layer)
        {
            Random rnd = new Random();

            for (int i = 0; i < (layer.weight.GetLength(0)); i++) //both for loops are used to loop through all elemets in the 2d array
            {
                for (int j = 0; j < (layer.weight.GetLength(1)); j++)
                {
                    layer.weight[i, j] = (2 * rnd.NextDouble()) - 1;  //sets a random double value for each weight
                }
            }

            for (int i = 0; i < (layer.bias.Length); i++)    //loops through the bias array
            {
                layer.bias[i] = (2 * rnd.NextDouble()) - 1;   //sets random double value for each bias
            }
        }
    }
}
