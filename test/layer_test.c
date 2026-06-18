//
// Created by User on 08.06.2026.
//

#include <stdlib.h>
#include <assert.h>
#include "../src/nn/common/layer/layer.h"

#include <time.h>

int inputs_count_create_layer = 50;
int nodes_count_create_layer = 50;
int biases_count_create_layer = 5;
int inputs_count_layer_set_inputs = 10;
int nodes_count_layer_set_inputs = 10;
int biases_count_layer_set_inputs = 3;
int full_inputs_count_layer_set_inputs = 13;
int inputs_count_layer_calculate = 5;
int nodes_count_layer_calculate = 5;
int biases_count_layer_calculate = 2;
double inputs_layer_calculate[] = {0.01, 0.12, 0.79, 0.5, 0.33};
double weights_for_all_nodes_layer_calculate[5][7] =
{
    {0.01, 0.12, 0.77, 0.15, 0.23, 0.24, 0.88},
    {0.03, 0.12, 0.76, 0.25, 0.33, 0.25, 0.87},
    {0.04, 0.12, 0.75, 0.35, 0.43, 0.26, 0.86},
    {0.05, 0.12, 0.74, 0.45, 0.53, 0.27, 0.85},
    {0.06, 0.12, 0.73, 0.55, 0.63, 0.28, 0.82}
};

double output_layer_sigma_calculate[] =
{
    0.86917682689316155820,
    0.87750366276423008571,
    0.88536024781276267959,
    0.89277450613545983327,
    0.89794528082384594825
};

double output_layer_tanh_calculate[] =
{
    0.95569485397923659153,
    0.96177068029667533988,
    0.96702085406649063160,
    0.97156045468873464266,
    0.97449516525259438637
};

double inputs[] = {0.01, 0.12, 0.79, 0.5, 0.33, 0.2, 0.8, 0.4, 0.6, 0.3};

void assert_create_layer(int inputCount, int nodesCount, int biasesCount) {
    t_layer *layer = malloc(sizeof(t_layer));
    layer_init(layer, inputCount, nodesCount, biasesCount);
    assert(inputs_count_create_layer == layer->inputsCount);
    assert(nodes_count_create_layer == layer->nodesCount);
    assert(biases_count_create_layer == layer->biasesCount);
    layer_destroy(layer);
    free(layer);
}

void assert_layer_set_inputs(int inputCount, int nodesCount, int biasesCount, double *inputs) {
    t_layer *layer = malloc(sizeof(t_layer));
    layer_init(layer, inputCount, nodesCount, biasesCount);
    layer_set_inputs(layer, inputs);
    for (int inputIndex = 0; inputIndex < layer->inputsCount; inputIndex++) {
        assert(inputs[inputIndex] == layer->inputs_buffer[inputIndex]);
    }
    for (int inputIndex = inputCount; inputIndex < layer->fullInputCount; inputIndex++) {
        assert(BIAS_DEFAULT_VALUE == layer->inputs_buffer[inputIndex]);
    }
    assert(full_inputs_count_layer_set_inputs == layer->fullInputCount);
    layer_destroy(layer);
    free(layer);
}

void assert_layer_calculate(int inputCount, int nodesCount, int biasesCount, double *inputs, double (*weights)[7]) {
    t_layer *layer = malloc(sizeof(t_layer));
    layer_init(layer, inputCount, nodesCount, biasesCount);
    layer_set_inputs(layer, inputs);

    for (int nodeIndex = 0; nodeIndex < layer->nodesCount; ++nodeIndex) {
        for (int inputIndex = 0; inputIndex < layer->nodes[nodeIndex].inputCount; ++inputIndex) {
            layer->nodes[nodeIndex].weights[inputIndex] = weights[nodeIndex][inputIndex];
        }
    }

    layer_calculate_sigma_output(layer);
    for (int nodeIndex = 0; nodeIndex < layer->nodesCount; ++nodeIndex) {
        assert(output_layer_sigma_calculate[nodeIndex] == layer->output[nodeIndex]);
    }

    layer_calculate_tanh_output(layer);
    for (int nodeIndex = 0; nodeIndex < layer->nodesCount; ++nodeIndex) {
        assert(output_layer_tanh_calculate[nodeIndex] == layer->output[nodeIndex]);
    }

    for (int inputIndex = 0; inputIndex < layer->inputsCount; inputIndex++) {
        assert(inputs[inputIndex] == layer->inputs_buffer[inputIndex]);
    }
    for (int inputIndex = inputCount; inputIndex < layer->fullInputCount; inputIndex++) {
        assert(BIAS_DEFAULT_VALUE == layer->inputs_buffer[inputIndex]);
    }

    layer_destroy(layer);
    free(layer);
}

int main() {
    srand(time(NULL));
    assert_create_layer(inputs_count_create_layer, nodes_count_create_layer, biases_count_create_layer);
    assert_layer_set_inputs(inputs_count_layer_set_inputs, nodes_count_layer_set_inputs, biases_count_layer_set_inputs,
        inputs);
    assert_layer_calculate(inputs_count_layer_calculate, nodes_count_layer_calculate, biases_count_layer_calculate,
                           inputs_layer_calculate, weights_for_all_nodes_layer_calculate);
    return 0;
}
