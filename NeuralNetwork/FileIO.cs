using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace NeuralNetwork
{
    public class FileIO
    {
        /// <summary>
        /// get input or output from a file
        /// </summary>
        /// <param name="filePath"></param>
        /// <returns></returns>
        public static List<double[]> readInputFromFile(string filePath)
        {
            StreamReader file = new StreamReader(filePath); //open file should be a csv file
            List<double[]> lst = new List<double[]>();  //create list to store the numbers
            string line;    //stores the current line fo text
            do
            {
                line = file.ReadLine(); //read line of file
                lst.Add(Array.ConvertAll(line.Split(','), Double.Parse));   //parse to doubles, from the csv
            } while (file.Peek() != -1);    //checks if there is more lines

            file.Close();
            return lst;
        }

        //FILE OUTPUT STRUCTURE .csv format
        //list of nodes per layer
        //goes through each layer starting in first hiden layer
        //lists bias values
        //lists weight values

        /// <summary>
        /// outputs the networks biases and weights to a file so that it can
        /// be loaded at a later time
        /// </summary>
        /// <param name="filePath"></param>
        /// <param name="n"></param>
        public static void outputNetwork(string filePath, Network n)
        {
            StreamWriter writer = new StreamWriter(filePath);

            string temp = "";
            foreach (Layer l in n.layers)
            {
                temp += l.neuron.Length.ToString() + ",";
            }
            writer.WriteLine(temp); //writes the first line with the list of nodes per layer

            for (int lNum = 1; lNum < n.layers.Count; lNum++)    //loop through each layer starting with hidden
            {
                Layer l = n.layers[lNum];
                temp = "";
                foreach (double b in l.bias) //loop through the biases
                {
                    temp += b.ToString() + ",";
                }
                writer.WriteLine(temp); //write to file

                for (int i = 0; i < l.weight.GetLength(0); i++)  //loop through first index of weight
                {
                    temp = "";
                    for (int j = 0; j < l.weight.GetLength(1); j++)  //loop through second index of weight
                    {
                        temp += l.weight[i, j].ToString() + ",";
                    }
                    writer.WriteLine(temp); //write weight to file
                }
            }

            writer.Close();
        }

        /// <summary>
        /// read file and load pregenerated neural network
        /// </summary>
        /// <param name="filePath"></param>
        /// <returns></returns>
        public static Network inputNetwork(string filePath)
        {
            StreamReader reader = new StreamReader(filePath);
            List<int> numNodes = new List<int>();
            int[] temp = Array.ConvertAll(reader.ReadLine().Split(','), Int32.Parse);   //read list of nodes
            numNodes = temp.ToList();

            Network n = new Network(numNodes);  //create network

            for (int layerNum = 1; layerNum < n.layers.Count; layerNum++)  //loop through each layer starting with first hidden layer
            {
                n.layers[layerNum].bias = Array.ConvertAll(reader.ReadLine().Split(','), Double.Parse); //set biases

                for (int i = 0; i < n.layers[layerNum].weight.GetLength(0); i++) //loop through each first index of weight
                {
                    double[] temp1 = Array.ConvertAll(reader.ReadLine().Split(','), Double.Parse);  //read line for that first index
                    for (int j = 0; j < n.layers[layerNum].weight.GetLength(0); j++)    //loop through second index
                    {
                        n.layers[layerNum].weight[i, j] = temp1[j]; //set corresponding value based on input
                    }
                }
            }
            reader.Close();
            return n;
        }

    }
}
