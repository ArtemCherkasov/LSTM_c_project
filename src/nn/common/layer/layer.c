//
// Created by User on 30.05.2026.
//
#include "layer.h"

#include <stdlib.h>

struct Layer createLayer(int inputCount, int nodesCount) {
    struct Layer layer;
    layer.inputCount = inputCount;
    layer.nodesCount = nodesCount;
    layer.nodes = (struct Node *) malloc(sizeof(struct Node) * nodesCount);
    for (int nodeIndex = 0; nodeIndex < layer.nodesCount; ++nodeIndex) {
        layer.nodes[nodeIndex] = node_createNode(layer.inputCount);
    }
    return layer;
}