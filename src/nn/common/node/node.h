//
// Created by User on 29.05.2026.
//

#ifndef LSTM_C_PROJECT_NODE_H
#define LSTM_C_PROJECT_NODE_H

struct Node {
    int inputCount;
    double *inputs;
    double *weights;
    double tempWeigth;
    double sum;
    double nodeValue;
    double deltaOfNode; // dE_dOut*dOut_dNet
    double *deltaOfWeight;
    int sizeOfNode;
};

typedef struct Node Node;

Node node_createNode(int inputCount);
void node_generateWeights(Node node);
void node_removeNode(Node node);
void node_setInput(Node node, double *input);
#endif //LSTM_C_PROJECT_NODE_H
