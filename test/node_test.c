//
// Created by User on 08.06.2026.
//

#include <stdlib.h>
#include <assert.h>
#include "../src/nn/common/node/node.h"

#include <stdio.h>
#include <time.h>

double inputs[] = {0.01, 0.12, 0.79, 0.5, 0.33, 0.2, 0.8, 0.4, 0.6, 0.3};
double weights[] = {0.002, 0.29, 0.08, 0.106, 0.234, 0.02, 0.99, 0.31, 0.92, 0.03};
double inputs_for_sigma[] = {0.01, 0.12};
double weights_for_sigma[] = {0.002, 0.29};
double inputs_for_tanh[] = {0.01, 0.12};
double weights_for_tanh[] = {0.002, 0.29};
double current_answer_for_sigma = 0.50870412058795244636;
double current_answer_for_tanh = 0.03480593452261415005;

void assert_create_node(int inputCount, int sizeOfNode){
    t_node *node = malloc(sizeof(t_node));
    node_init(node, inputCount);
    assert(node->inputCount == inputCount);
    assert(node->sizeOfNode == sizeOfNode);
    node_destroy(node);
    free(node);
}

void assert_set_input_to_node(int inputCount, double *inputs){
    t_node *node = malloc(sizeof(t_node));
    node_init(node, inputCount);
    node_set_input(node, inputs);
    assert(node->inputs[0] == inputs[0]);
    assert(node->inputs[1] == inputs[1]);
    assert(node->inputs[7] == inputs[7]);
    node_destroy(node);
    free(node);
}

void assert_sigma_activate(int inputCount, double *inputs, double *weights){
    t_node *node = malloc(sizeof(t_node));
    node_init(node, inputCount);
    node_set_input(node, inputs);
    node_set_weights(node, weights);
    node_self_value_sigma_calculation(node);
    assert(node->nodeValue == current_answer_for_sigma);
    node_destroy(node);
    free(node);
}

void assert_tanh_activate(int inputCount, double *inputs, double *weights){
    t_node *node = malloc(sizeof(t_node));
    node_init(node, inputCount);
    node_set_input(node, inputs);
    node_set_weights(node, weights);
    node_self_value_tanh_calculation(node);
    assert(node->nodeValue == current_answer_for_tanh);
    node_destroy(node);
    free(node);
}

int main(){
    srand(time(NULL));
    assert_create_node(220, 6240);
    assert_set_input_to_node(10, inputs);
    assert_sigma_activate(2, inputs_for_sigma, weights_for_sigma);
    assert_tanh_activate(2, inputs_for_sigma, weights_for_sigma);
    return 0;
}