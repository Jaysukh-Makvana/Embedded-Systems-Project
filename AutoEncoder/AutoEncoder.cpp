#include <bits/stdc++.h>
#include "image.h"
#include "model_params.h"

using namespace std;

class AutoEncoder
{
public:
    // Constructor
    AutoEncoder()
    {
    }

    // Forward pass function
    vector<vector<vector<vector<long double>>>> forward(vector<vector<vector<vector<long double>>>> input)
    {
        // Apply Convolution on input
        vector<vector<vector<vector<long double>>>> conv1_out = conv2d(input, encoder_0_weight, encoder_0_bias, 1);
        relu(conv1_out);

        // Apply Convolution on conv1_out
        vector<vector<vector<vector<long double>>>> conv2_out = conv2d(conv1_out, encoder_2_weight, encoder_2_bias, 1);
        relu(conv2_out);

        // Apply Convolution on conv2_out
        vector<vector<vector<vector<long double>>>> convT1_out = conv2d(conv2_out, decoder_0_weight, decoder_0_bias, 1);
        relu(convT1_out);

        // Apply Convolution on convT1_out
        vector<vector<vector<vector<long double>>>> convT2_out = conv2d(convT1_out, decoder_2_weight, decoder_2_bias, 1);

        return convT2_out;
    }

private:
    // Helper function for 2D convolution
    vector<vector<vector<vector<long double>>>> conv2d(vector<vector<vector<vector<long double>>>> input,
                                                       vector<vector<vector<vector<long double>>>> weights,
                                                       vector<long double> bias, int padding)
    {
        int input_batch = input.size();
        int input_channel = input[0].size();
        int input_height = input[0][0].size();
        int input_width = input[0][0][0].size();

        int num_filters = weights.size();
        int filter_depth = weights[0].size();
        int filter_height = weights[0][0].size();
        int filter_width = weights[0][0][0].size();

        int output_height = input_height - filter_height + (2 * padding) + 1;
        int output_width = input_width - filter_width + (2 * padding) + 1;

        vector<vector<vector<vector<long double>>>> output(input_batch,
                                                           vector<vector<vector<long double>>>(num_filters,
                                                                                               vector<vector<long double>>(output_height,
                                                                                                                           vector<long double>(output_width, 0.0))));

        // Perform Convolution
        for (int b = 0; b < input_batch; b++)
        {
            for (int c = 0; c < input_channel; c++)
            {
                for (int f = 0; f < num_filters; f++)
                {
                    for (int h = 1; h < output_height + 2 * padding - 1; h++)
                    {
                        for (int w = 1; w < output_width + 2 * padding - 1; w++)
                        {
                            for (int i = 2 - filter_height; i < filter_height - 1; i++)
                            {
                                for (int j = 2 - filter_width; j < filter_width - 1; j++)
                                {
                                    if (h + i - 1 >= 0 && h + i - 1 < input_height && w + j - 1 >= 0 && w + j - 1 < input_width)
                                    {
                                        output[b][f][h - 1][w - 1] += (input[b][c][h + i - 1][w + j - 1] * weights[f][c][i + filter_height - 2][j + filter_width - 2]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Add Bias
        for (int i = 0; i < output.size(); i++)
        {
            for (int j = 0; j < output[0].size(); j++)
            {
                for (int k = 0; k < output[0][0].size(); k++)
                {
                    for (int l = 0; l < output[0][0][0].size(); l++)
                    {
                        output[i][j][k][l] += bias[j];
                    }
                }
            }
        }
        return output;
    }

    // Helper function for ReLU activation
    void relu(vector<vector<vector<vector<long double>>>> &input)
    {
        // Perform ReLU activation inplace
        for (auto &batch : input)
        {
            for (auto &channel : batch)
            {
                for (auto &row : channel)
                {
                    for (auto &val : row)
                    {
                        val = max(val, (long double)0.0);
                    }
                }
            }
        }
    }
};

int main()
{
    // Create an instance of the AutoEncoder model
    AutoEncoder model;

    // Initialize parameter
    int batch = 1;
    int channel = 3;
    int height = 32;
    int width = 32;

    // Create a dummy input tensor
    vector<vector<vector<vector<long double>>>> input = image;

    // Write input in File
    ofstream in_file;
    in_file.open("input.txt");
    for (auto i : input)
    {
        for (auto j : i)
        {
            for (auto k : j)
            {
                for (auto l : k)
                {
                    in_file << l << " ";
                }
            }
        }
    }
    in_file.close();

    // Perform forward pass
    vector<vector<vector<vector<long double>>>> output = model.forward(input);

    // Write output in File
    ofstream out_file;
    out_file.open("output.txt");
    for (auto i : output)
    {
        for (auto j : i)
        {
            for (auto k : j)
            {
                for (auto l : k)
                {
                    out_file << l << " ";
                }
                // out_file << "\n";
            }
            // out_file << "\n";
        }
        // out_file << "\n";
    }
    out_file.close();

    return 0;
}