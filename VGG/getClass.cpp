#include <bits/stdc++.h>
using namespace std;

vector<float> softmax(vector<float> x)
{
    vector<float> exp_x;
    for (int i = 0; i < x.size(); i++)
    {
        exp_x.push_back(exp(x[i]));
    }
    float sum = 0.0;
    for (int i = 0; i < exp_x.size(); i++)
    {
        sum += exp_x[i];
    }
    vector<float> result;
    for (int i = 0; i < exp_x.size(); i++)
    {
        result.push_back(exp_x[i] / sum);
    }

    return result;
}

int main()
{
    // Read the output tensor
    ifstream output_file("output.txt");
    vector<float> output;
    float value;
    while (output_file >> value)
    {
        output.push_back(value);
    }

    // Apply softmax function
    vector<float> probabilities = softmax(output);

    // Get the class with the highest probability
    int max_idx = 0;
    for (int i = 1; i < probabilities.size(); i++)
    {
        if (probabilities[i] > probabilities[max_idx])
        {
            max_idx = i;
        }
    }

    cout << "Predicted Class: " << max_idx << "\n";
}