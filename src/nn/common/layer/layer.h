//
// Created by User on 30.05.2026.
//

#ifndef LSTM_C_PROJECT_LAYER_H
#define LSTM_C_PROJECT_LAYER_H
#include "../node/node.h"
#include "../bias/bias.h"

typedef struct Layer t_layer;

struct Layer {
    int inputsCount;
    int nodesCount;
    int biasesCount;
    int fullInputCount;
    t_node *nodes;
    t_bias *biases;
    double *inputs_buffer;
    double *output;
};

void layer_init(t_layer *layer, int inputsCount, int nodesCount, int biasesCount);
void layer_destroy(t_layer *layer);
void layer_calculate_sigma_output(t_layer *layer);
void layer_calculate_tanh_output(t_layer *layer);
void layer_set_inputs(t_layer *layer, double *inputs);
#endif //LSTM_C_PROJECT_LAYER_H

