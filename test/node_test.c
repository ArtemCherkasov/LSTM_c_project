//
// Created by User on 08.06.2026.
//

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <stdbool.h>
#include "../src/nn/common/node/node.h"

double _inputs[] = {0.01, 0.12, 0.79, 0.5, 0.33, 0.2, 0.8, 0.4, 0.6, 0.3};
double _weights[] = {0.002, 0.29, 0.08, 0.106, 0.234, 0.02, 0.99, 0.31, 0.92, 0.03};
double _inputs_for_sigma[] = {0.01, 0.12};
double _weights_for_sigma[] = {0.002, 0.29};
double _current_answer_for_sigma = 0.50870412058795244636;

void assert_create_node(int inputCount, int sizeOfNode){
    t_node *node = malloc(sizeof(t_node));
    node_init(node, inputCount);
    assert(node->inputCount == inputCount);
    assert(node->sizeOfNode == sizeOfNode);
    node_destroy(node);
}

void assert_set_input_to_node(int inputCount, double *inputs){
    t_node *node = malloc(sizeof(t_node));
    node_init(node, inputCount);
    node_set_input(node, inputs);
    assert(node->inputs[0] == inputs[0]);
    assert(node->inputs[1] == inputs[1]);
    assert(node->inputs[7] == inputs[7]);
    node_destroy(node);
}

void assert_sigma_activate(int inputCount, double *inputs, double *weights){
    t_node *node = malloc(sizeof(t_node));
    node_init(node, inputCount);
    node_set_input(node, inputs);
    node_set_weights(node, weights);
    node_self_value_sigma_calculation(node);
    assert(node->nodeValue == _current_answer_for_sigma);
    node_destroy(node);
}

int main(){
    assert_create_node(220, 6240);
    assert_set_input_to_node(10, _inputs);
    assert_sigma_activate(2, _inputs_for_sigma, _weights_for_sigma);
    return 0;
}