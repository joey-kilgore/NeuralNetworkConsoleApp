using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    class Program
    {
        static void Main(string[] args)
        {
            List<int> networkNodeList = new List<int>() { 1, 50, 50, 1 };
            Network network = new Network(networkNodeList);

            double[] finalNeurons = network.calcOutput(new double[] { .8 });
            Console.WriteLine("TEST ORIGINAL VALUE : " + finalNeurons[0]);
            //network.rawOutput();


            Console.WriteLine("TRY UPDATE");
            sinTest(network);

            Console.Read();
        }

        public static void sinTest(Network network)
        {
            double[] finalNeurons;
            List<double[]> inputStuff = FileIO.readInputFromFile(@"C: \Users\joeya\Desktop\Neural Network\sinInput.csv");
            List<double[]> outputStuff = FileIO.readInputFromFile(@"C: \Users\joeya\Desktop\Neural Network\sinOutput.csv");

            for (int i = 0; i < 100; i++)
            {
                network.train(inputStuff, outputStuff, .5, 1000);

                finalNeurons = network.calcOutput(new double[] { .8 });
                Console.WriteLine("TEST " + i + " VALUE : " + finalNeurons[0]);
            }
            Console.ReadLine();
        }
    }
}
