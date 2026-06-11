//
// Created by User on 29.05.2026.
//
#include "node.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>

void node_init(t_node *node, int inputCount) {
    node->inputCount = inputCount;
    node->inputs = malloc(sizeof(double) * inputCount);
    node->weights = malloc(sizeof(double) * inputCount);
    node->deltaOfWeight = malloc(sizeof(double) * inputCount);
    node->directions = malloc(sizeof(t_direction) * inputCount);
    node->sum = 0.0;
    node->nodeValue = 0.0;
    node->sizeOfNode = sizeof(struct Node) + sizeof(double) * inputCount * 3 + sizeof(t_direction) * inputCount;
    node_generate_weights(node);
    node_init_directions(node);
}

void node_generate_weights(t_node *node) {
    srand(time(NULL));
    for (int weightIndex = 0; weightIndex < node->inputCount; ++weightIndex) {
        node->weights[weightIndex] = (double) rand() / (double) RAND_MAX;
    }
}

void node_set_weights(t_node *node, double *weigths) {
    memcpy(node->weights, weigths, node->inputCount * sizeof(double));
}

void node_set_input(t_node *node, double *input) {
    memcpy(node->inputs, input, node->inputCount * sizeof(double));
}

void node_self_sum_calculation(t_node *node) {
    double sum = 0.0;
    for (int inputIndex = 0; inputIndex < node->inputCount; ++inputIndex) {
        sum = sum + node->inputs[inputIndex] * node->weights[inputIndex];
    }
    node->sum = sum;
}

double node_sigma_activation(double summ) {
    return 1.0 / (1.0 + exp(-summ));
}

double node_tanh_activation(double summ) {
    double exp_pow_2_summ = exp(2.0 * summ);
    return (exp_pow_2_summ - 1.0) / (exp_pow_2_summ + 1.0);
}

void node_self_value_sigma_calculation(t_node *node) {
    node_self_sum_calculation(node);
    node->nodeValue = node_sigma_activation(node->sum);
}

void node_self_value_tanh_calculation(t_node *node) {
    node_self_sum_calculation(node);
    node->nodeValue = node_tanh_activation(node->sum);
}

void node_destroy(t_node *node) {
    free(node->inputs);
    free(node->weights);
    free(node->deltaOfWeight);
}

void node_set_change_action(t_node *node, int weight_index, t_direction direction) {

}

void node_init_directions(t_node *node) {
    for (int directionIndex = 0; directionIndex < node->inputCount; ++directionIndex) {
        node->directions[directionIndex] = IMMUTABLE;
    }
}
