#include <bits/stdc++.h>
#include "image.h"
#include "model_params.h"

using namespace std;

class MiniResNet
{
public:
    // Constructor
    MiniResNet()
    {
    }

    // Forward pass function
    vector<vector<long double>> forward(vector<vector<vector<vector<long double>>>> input)
    {
        // Residual block 1
        vector<vector<vector<vector<long double>>>> conv1_out = conv2d(input, conv1_weight, conv1_bias, 1);
        vector<vector<vector<vector<long double>>>> batchNorm1_out = batchNorm2d(conv1_out, bn1_weight, bn1_bias);
        relu(batchNorm1_out);

        // Residual block 2
        vector<vector<vector<vector<long double>>>> conv2_out = conv2d(batchNorm1_out, conv2_weight, conv2_bias, 1);
        vector<vector<vector<vector<long double>>>> batchNorm2_out = batchNorm2d(conv2_out, bn2_weight, bn2_bias);
        relu(batchNorm2_out);

        // Residual block 3
        vector<vector<vector<vector<long double>>>> conv3_out = conv2d(batchNorm2_out, conv3_weight, conv3_bias, 1);
        vector<vector<vector<vector<long double>>>> batchNorm3_out = batchNorm2d(conv3_out, bn3_weight, bn3_bias);

        // Addition of batchNorm3_out and batchNorm1_out
        vector<vector<vector<vector<long double>>>> res1_output = vector_sum(batchNorm3_out, batchNorm1_out);
        relu(res1_output);

        // Apply Max Pooling
        vector<vector<vector<vector<long double>>>> max_pool_output = max_pool(res1_output);

        // Apply Convolution
        vector<vector<vector<vector<long double>>>> conv4_output = conv2d(max_pool_output, conv4_weight, conv4_bias, 1);

        // Apply Flatten Operation
        vector<vector<long double>> flatten_output = flatten(conv4_output);

        // Apply Linear Layer
        vector<vector<long double>> linear_output = matmul(flatten_output, transpose(fc_weight), fc_bias);

        return linear_output;
    }

private:
    // Helper function for matrix transpose
    vector<vector<long double>> transpose(vector<vector<long double>> input)
    {
        vector<vector<long double>> output(input[0].size(), vector<long double>(input.size(), 0.0));
        for (int i = 0; i < input.size(); i++)
        {
            for (int j = 0; j < input[0].size(); j++)
            {
                output[j][i] = input[i][j];
            }
        }
        return output;
    }

    // Helper function for matrix multiplication
    vector<vector<long double>> matmul(vector<vector<long double>> input, vector<vector<long double>> weights, vector<long double> bias)
    {
        vector<vector<long double>> output(input.size(), vector<long double>(bias.size(), 0.0));
        for (int i = 0; i < input.size(); i++)
        {
            for (int j = 0; j < weights[0].size(); j++)
            {
                for (int k = 0; k < input[0].size(); k++)
                {
                    output[i][j] += (input[i][k] * weights[k][j]);
                }
                output[i][j] += bias[j];
            }
        }
        return output;
    }

    // Helper function for flattening the tensor
    vector<vector<long double>> flatten(vector<vector<vector<vector<long double>>>> input)
    {
        vector<vector<long double>> output;
        for (int b = 0; b < input.size(); ++b)
        {
            vector<long double> temp;
            for (int c = 0; c < input[0].size(); ++c)
            {
                for (int h = 0; h < input[0][0].size(); ++h)
                {
                    for (int w = 0; w < input[0][0][0].size(); ++w)
                    {
                        temp.push_back(input[b][c][h][w]);
                    }
                }
            }
            output.push_back(temp);
        }
        return output;
    }

    // Helper function for element-wise vector addition
    vector<vector<vector<vector<long double>>>> vector_sum(vector<vector<vector<vector<long double>>>> input1,
                                                           vector<vector<vector<vector<long double>>>> input2)
    {
        // Perform element-wise vector addition
        for (int b = 0; b < input1.size(); ++b)
        {
            for (int c = 0; c < input1[0].size(); ++c)
            {
                for (int h = 0; h < input1[0][0].size(); ++h)
                {
                    for (int w = 0; w < input1[0][0][0].size(); ++w)
                    {
                        input1[b][c][h][w] += input2[b][c][h][w];
                    }
                }
            }
        }
        return input1;
    }

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

    // Helper function for batch normalization
    vector<vector<vector<vector<long double>>>> batchNorm2d(vector<vector<vector<vector<long double>>>> input,
                                                            vector<long double> weight,
                                                            vector<long double> bias)
    {
        // Calculate running mean and variance
        vector<long double> running_mean(input[0].size(), 0.0);
        vector<long double> running_var(input[0].size(), 0.0);

        // Calculate running Mean
        for (int c = 0; c < input[0].size(); ++c)
        {
            for (int b = 0; b < input.size(); ++b)
            {
                for (int h = 0; h < input[0][0].size(); ++h)
                {
                    for (int w = 0; w < input[0][0][0].size(); ++w)
                    {
                        running_mean[c] += input[b][c][h][w];
                    }
                }
            }
            running_mean[c] /= input.size() * input[0][0].size() * input[0][0][0].size();
        }

        // Calculate running Variance
        for (int c = 0; c < input[0].size(); ++c)
        {
            for (int b = 0; b < input.size(); ++b)
            {
                for (int h = 0; h < input[0][0].size(); ++h)
                {
                    for (int w = 0; w < input[0][0][0].size(); ++w)
                    {
                        running_var[c] += pow(input[b][c][h][w] - running_mean[c], 2);
                    }
                }
            }
            running_var[c] /= input.size() * input[0][0].size() * input[0][0][0].size();
        }

        // Perform batch normalization
        for (int b = 0; b < input.size(); ++b)
        {
            for (int c = 0; c < input[0].size(); ++c)
            {
                for (int h = 0; h < input[0][0].size(); ++h)
                {
                    for (int w = 0; w < input[0][0][0].size(); ++w)
                    {
                        input[b][c][h][w] = (input[b][c][h][w] - running_mean[c]) / sqrt(running_var[c] + 1e-5);
                        input[b][c][h][w] = input[b][c][h][w] * weight[c] + bias[c];
                    }
                }
            }
        }
        return input;
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

    // Helper function for max pooling
    vector<vector<vector<vector<long double>>>> max_pool(vector<vector<vector<vector<long double>>>> input)
    {
        // Perform max pooling operation
        vector<vector<vector<vector<long double>>>> output(input.size(),
                                                           vector<vector<vector<long double>>>(input[0].size(), vector<vector<long double>>(input[0][0].size() / 2, vector<long double>(input[0][0][0].size() / 2, 0))));
        for (int b = 0; b < input.size(); ++b)
        {
            for (int c = 0; c < input[0].size(); ++c)
            {
                for (int h = 0; h < input[0][0].size(); h += 2)
                {
                    for (int w = 0; w < input[0][0][0].size(); w += 2)
                    {
                        long double max_val = numeric_limits<long double>::lowest();
                        max_val = max(max_val, input[b][c][h][w]);
                        max_val = max(max_val, input[b][c][h][w + 1]);
                        max_val = max(max_val, input[b][c][h + 1][w]);
                        max_val = max(max_val, input[b][c][h + 1][w + 1]);
                        output[b][c][h / 2][w / 2] = max_val;
                    }
                }
            }
        }
        return output;
    }
};

int main()
{
    // Create an instance of the MiniResNet model
    MiniResNet model;

    // Initialize parameter
    int batch = 1;
    int channel = 3;
    int height = 32;
    int width = 32;

    // Create a dummy input tensor
    vector<vector<vector<vector<long double>>>> input = image;

    // Perform forward pass
    vector<vector<long double>> output = model.forward(input);

    // Write output in File
    ofstream out_file;
    out_file.open("output.txt");
    for (auto i : output)
    {
        for (auto j : i)
        {
            out_file << j << " ";
        }
        out_file << "\n";
    }
    out_file.close();

    return 0;
}