//
// Created by User on 30.05.2026.
//

#ifndef LSTM_C_PROJECT_LAYER_H
#define LSTM_C_PROJECT_LAYER_H
#include "../node/node.h"
#include "../bias/bias.h"

struct Layer {
    int inputCount;
    int nodesCount;
    struct Node *nodes;
    struct Bias *biases;
};

struct Layer createLayer(int inputCount, int nodesCount);
#endif //LSTM_C_PROJECT_LAYER_H

