//
// Created by User on 29.05.2026.
//

#ifndef LSTM_C_PROJECT_NODE_H
#define LSTM_C_PROJECT_NODE_H

typedef struct Node t_node;
typedef enum Direction t_direction;

struct Node {
    int inputCount;
    double *inputs;
    double *weights;
    double tempWeigth;
    double sum;
    double nodeValue;
    double deltaOfNode; // dE_dOut*dOut_dNet
    double *deltaOfWeight;
    t_direction *directions;
    int sizeOfNode;
};

enum Direction {
    NEGATIVE=1,
    POSITIVE=2,
    IMMUTABLE=3
};

void node_init(t_node *node, int inputCount);

void node_generate_weights(t_node *node);

void node_set_weights(t_node *node, double *weigths);

void node_destroy(t_node *node);

void node_set_input(t_node *node, double *input);

double node_sigma_activation(double summ);

void node_self_sum_calculation(t_node *node);

void node_self_value_sigma_calculation(t_node *node);

void node_self_value_tanh_calculation(t_node *node);

void node_set_direction(t_node *node, int weight_index, t_direction direction);

void node_action_with_saving_weight(t_node *node, int weight_index, double weight_delta_value);

void node_action_simple(t_node *node, int weight_index, double weight_delta_value);

void node_weight_recover(t_node *node, int weight_index);

void node_init_directions(t_node *node);
#endif //LSTM_C_PROJECT_NODE_H
