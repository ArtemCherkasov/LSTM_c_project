//
// Created by User on 30.05.2026.
//

#ifndef LSTM_C_PROJECT_LAYER_H
#define LSTM_C_PROJECT_LAYER_H
#include "../node/node.h"
#include "../bias/bias.h"

typedef struct Layer t_layer;

struct Layer {
    int inputCount;
    int nodesCount;
    int biasesCount;
    t_node *nodes;
    t_bias *biases;
};

void layer_init(t_layer *layer, int inputCount, int nodesCount, int biasesCount);
void layer_destroy(t_layer *layer);
#endif //LSTM_C_PROJECT_LAYER_H

