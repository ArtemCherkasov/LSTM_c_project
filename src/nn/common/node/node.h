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

struct Node createNode(int inputCount);
void removeNode(struct Node node);
#endif //LSTM_C_PROJECT_NODE_H
