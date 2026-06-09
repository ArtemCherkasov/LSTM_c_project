//
// Created by User on 08.06.2026.
//

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <stdbool.h>
#include "../src/nn/common/node/node.h"

double _inputs[] = {0.01, 0.12, 0.79, 0.5, 0.33, 0.2, 0.8, 0.4, 0.6, 0.3};

void assert_create_node(int inputCount, int sizeOfNode){
    struct Node node = node_createNode(inputCount);
    assert(node.inputCount == inputCount);
    assert(node.sizeOfNode == sizeOfNode);
    node_removeNode(node);
}

void assert_set_input_to_node(int inputCount, double *inputs){
    struct Node node = node_createNode(inputCount);
    node_setInput(node, inputs);
    assert(node.inputs[0] == inputs[0]);
    assert(node.inputs[1] == inputs[1]);
    assert(node.inputs[7] == inputs[7]);
    node_removeNode(node);
}

int main(){
    assert_create_node(220, 5352);
    assert_set_input_to_node(10, _inputs);
    return 0;
}