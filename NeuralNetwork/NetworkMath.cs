using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class NetworkMath
    {
        public double[] multiply(double[,] a, double[] b)    //2d matrix times a 1d matrix
        {
            double[] c = new double[a.Length];  //instantiates the correct size of the product matrix

            for (int i = 0; i < (a.GetLength(0)); i++)   //loop through each row in the 2d array
            {
                double sum = 0;
                for (int j = 0; j < (a.GetLength(1)); j++)   //loop through each element of the row 
                {
                    sum += a[i, j] * b[j]; //calculates the sum of the product between the two respective elements from both matrices
                }
                c[i] = sum; //sets the respective value for the matrix to be returned
            }

            return c;
        }

        public double[] add(double[] a, double[] b)  //add 2 1d matrices
        {
            double[] c = new double[a.Length];  //instantiates the correct size of the summation matrix

            for (int i = 0; i < (a.Length); i++)
            {
                c[i] = a[i] + b[i];
            }

            return c;
        }

        public double[,] add(double[,] a, double[,] b)
        {
            double[,] c = new double[a.GetLength(0), a.GetLength(1)];
            for (int i = 0; i < a.GetLength(0); i++)
            {
                for (int j = 0; j < a.GetLength(1); j++)
                {
                    c[i, j] = a[i, j] + b[i, j];
                }
            }
            return c;
        }

        public double[] activationFunction(double[] a)   //applies activation function to each element
        {
            double[] temp = new double[a.Length];
            for (int i = 0; i < (a.Length); i++)
            {
                temp[i] = (1.0) / (1.0 + Math.Exp(-1.0 * a[i])); //logistic function
            }

            return temp;
        }

        /// <summary>
        /// required to calculated the value of a neuron
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        public double activationFunctionDeriv(double a)
        {
            return (Math.Exp(a)) / Math.Pow((1.0 + Math.Exp(a)), 2.0);
        }

        /// <summary>
        /// required to calculate the values during back propogation
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        public double[] activationFunctionDeriv(double[] a)
        {
            double[] temp = new double[a.Length];
            for (int i = 0; i < a.Length; i++)
            {
                temp[i] = (Math.Exp(a[i])) / Math.Pow((1.0 + Math.Exp(a[i])), 2.0);
            }
            return temp;
        }
    }
}
