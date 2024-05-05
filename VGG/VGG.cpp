#include <bits/stdc++.h>
#include "image.h"
#include "model_params.h"

using namespace std;

// Define the VGG model class
class VGG
{
public:
    // Constructor
    VGG()
    {
    }

    // Forward pass function
    vector<long double> forward(vector<vector<vector<vector<long double>>>> input)
    {
        vector<vector<vector<vector<long double>>>> conv1_out = conv2d(input, features_0_weight, features_0_bias, 0);
        relu(conv1_out);

        vector<vector<vector<vector<long double>>>> conv2_out = conv2d(conv1_out, features_2_weight, features_2_bias, 0);
        relu(conv2_out);

        vector<vector<vector<vector<long double>>>> pool1_out = max_pool(conv2_out);

        vector<vector<vector<vector<long double>>>> conv3_out = conv2d(pool1_out, features_5_weight, features_5_bias, 0);
        relu(conv3_out);

        vector<vector<vector<vector<long double>>>> conv4_out = conv2d(conv3_out, features_7_weight, features_7_bias, 0);
        relu(conv4_out);

        vector<vector<vector<vector<long double>>>> pool2_out = max_pool(conv4_out);

        vector<vector<vector<vector<long double>>>> conv5_out = conv2d(pool2_out, classifier_0_weight, classifier_0_bias, 0);
        relu(conv5_out);

        vector<long double> pool3_out = adaptiveAvgPool2d(conv5_out, 1, 1);

        return pool3_out;
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

    vector<long double> adaptiveAvgPool2d(vector<vector<vector<vector<long double>>>> &input, int output_height, int output_width)
    {
        int input_batch = input.size();
        int input_channel = input[0].size();
        int input_height = input[0][0].size();
        int input_width = input[0][0][0].size();

        vector<long double> output(input_batch * input_channel);

        int idx = 0;
        for (int b = 0; b < input_batch; b++)
        {
            for (int c = 0; c < input_channel; c++)
            {
                long double sum = 0;
                for (int i = 0; i < input_height; i++)
                {
                    for (int j = 0; j < input_width; j++)
                    {
                        sum += input[b][c][i][j];
                    }
                }
                output[idx] = sum / (input_height * input_width);
                idx++;
            }
        }
        return output;
    }
};

int main()
{
    // Create an instance of the VGG model
    VGG model;

    int height = 32;
    int width = 32;
    int channels = 1;
    int batch_size = 1;
    int input_height = 4;
    int input_width = 4;

    vector<vector<vector<vector<long double>>>> input = image;

    // Perform forward pass
    vector<long double> output = model.forward(input);

    // Write output in File
    ofstream out_file;
    out_file.open("output.txt");
    for (auto &val : output)
    {
        out_file << val << " ";
    }
    out_file << "\n";
    out_file.close();

    return 0;
}
