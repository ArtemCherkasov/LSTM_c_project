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

typedef struct Node t_node;

void node_init(t_node *node, int inputCount);
void node_generate_weights(t_node *node);
void node_set_weights(t_node *node, double *weigths);
void node_destroy(t_node *node);
void node_set_input(t_node *node, double *input);
double node_sigma_activation(double summ);
double node_tanh_activation(double summ);
void node_self_sum_calculation(t_node *node);
void node_self_value_sigma_calculation(t_node *node);
void node_self_value_tanh_calculation(t_node *node);
#endif //LSTM_C_PROJECT_NODE_H
