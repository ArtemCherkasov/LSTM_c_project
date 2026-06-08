//
// Created by User on 29.05.2026.
//
#include "node.h"

#include <stdlib.h>
#include <stdio.h>

struct Node createNode(int inputCount) {
    struct Node node;
    node.inputCount = inputCount;
    node.inputs = malloc(sizeof(double) * inputCount);
    node.weights = malloc(sizeof(double) * inputCount);
    node.deltaOfWeight = malloc(sizeof(double) * inputCount);
    node.sizeOfNode = sizeof(struct Node) + sizeof(double) * inputCount * 3;
    return node;
}

void removeNode(struct Node node){
    free(node.inputs);
    free(node.weights);
    free(node.deltaOfWeight);
}