//
// Created by User on 29.05.2026.
//
#include "node.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

Node node_createNode(int inputCount) {
    Node node;
    node.inputCount = inputCount;
    node.inputs = malloc(sizeof(double) * inputCount);
    node.weights = malloc(sizeof(double) * inputCount);
    node.deltaOfWeight = malloc(sizeof(double) * inputCount);
    node.sum = 0.0;
    node.sizeOfNode = sizeof(struct Node) + sizeof(double) * inputCount * 3;
    node_generateWeights(node);
    return node;
}

void node_generateWeights(Node node) {
    srand(time(NULL));
    for (int weightIndex = 0; weightIndex < node.inputCount; ++weightIndex) {
        node.weights[weightIndex] = (double) rand() / (double) RAND_MAX;
    }
}

void node_setInput(Node node, double *input){
    memcpy(node.inputs, input, node.inputCount * sizeof (double ));
}

void node_removeNode(Node node) {
    free(node.inputs);
    free(node.weights);
    free(node.deltaOfWeight);
}