//
// Created by User on 30.05.2026.
//
#include "layer.h"

#include <stdlib.h>

void layer_init(t_layer *layer, int inputCount, int nodesCount, int biasesCount) {
    layer->inputCount = inputCount;
    layer->nodesCount = nodesCount;
    layer->biasesCount = biasesCount;
    layer->nodes = (t_node *) malloc(sizeof(t_node) * nodesCount);
    layer->biases = (t_bias *) malloc(sizeof(t_bias) * biasesCount);
    for (int nodeIndex = 0; nodeIndex < layer->nodesCount; ++nodeIndex) {
        node_init(&layer->nodes[nodeIndex], inputCount + biasesCount);
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
}