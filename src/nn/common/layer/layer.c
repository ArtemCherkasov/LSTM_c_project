//
// Created by User on 30.05.2026.
//
#include "layer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void layer_init(t_layer *layer, int inputsCount, int nodesCount, int biasesCount) {
    layer->inputsCount = inputsCount;
    layer->nodesCount = nodesCount;
    layer->biasesCount = biasesCount;
    layer->fullInputCount = inputsCount + biasesCount;
    layer->nodes = (t_node *) malloc(sizeof(t_node) * nodesCount);
    layer->biases = (t_bias *) malloc(sizeof(t_bias) * biasesCount);
    layer->inputs_buffer = (double *) malloc(sizeof(double) * (inputsCount + biasesCount));
    layer->output = malloc(sizeof(double) * nodesCount);
    for (int nodeIndex = 0; nodeIndex < layer->nodesCount; ++nodeIndex) {
        node_init(&layer->nodes[nodeIndex], inputsCount + biasesCount);
    }
    for (int biasIndex = 0; biasIndex < layer->biasesCount; ++biasIndex) {
        bias_init(&layer->biases[biasIndex]);
    }
}

void layer_destroy(t_layer *layer) {
    for (int nodeIndex = 0; nodeIndex < layer->nodesCount; ++nodeIndex) {
        node_destroy(&layer->nodes[nodeIndex]);
    }
    free(layer->nodes);
    free(layer->biases);
    free(layer->inputs_buffer);
    free(layer->output);
}

void layer_set_inputs(t_layer *layer, double *inputs) {
    memcpy(layer->inputs_buffer, inputs, layer->inputsCount * sizeof(double));
    for (int inputIndex = layer->inputsCount; inputIndex < layer->fullInputCount; ++inputIndex) {
        layer->inputs_buffer[inputIndex] = layer->biases[-layer->fullInputCount + layer->biasesCount + inputIndex].biasValue;
    }
    for (int nodeIndex = 0; nodeIndex < layer->nodesCount; ++nodeIndex) {
        node_set_input(&layer->nodes[nodeIndex], layer->inputs_buffer);
    }
}

void layer_calculate_sigma_output(t_layer *layer) {
    for (int nodeIndex = 0; nodeIndex < layer->nodesCount; ++nodeIndex) {
        node_self_value_sigma_calculation(&layer->nodes[nodeIndex]);
        layer->output[nodeIndex] = layer->nodes[nodeIndex].nodeValue;
    }
}

void layer_calculate_tanh_output(t_layer *layer) {
    for (int nodeIndex = 0; nodeIndex < layer->nodesCount; ++nodeIndex) {
        node_self_value_tanh_calculation(&layer->nodes[nodeIndex]);
        layer->output[nodeIndex] = layer->nodes[nodeIndex].nodeValue;
    }
}