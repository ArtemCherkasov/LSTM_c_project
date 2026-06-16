//
// Created by User on 08.06.2026.
//

#include "../src/nn/lstm/lstm_cell/lstm_cell.h"

#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

#include "../src/nn/common/layer/layer.h"

#include <time.h>

int inputs_count_create_cell = 10;
int nodes_count_create_cell = 11;

double hidden_state_inputs[] = {0.0801, -0.0812, -0.0879, 0.085, -0.0833, 0.082, -0.088, 0.084, -0.086, 0.083, -0.0891};
double cell_state_inputs[] = {0.101, 0.112, 0.179, 0.185, 0.133, 0.12, 0.18, 0.14, 0.16, 0.13, 0.191};
double inputs[] = {0.01, 0.12, -0.79, 0.5, 0.33, 0.2, -0.8, 0.4, 0.6, 0.3};
double weights_for_all_nodes_forget_gate[11][22] =
{
    {0.65, 0.60, 0.76, 0.49, 0.21, 0.59, 0.09, 0.63, 0.94, 0.43, 0.36, 0.34, 0.55, 0.52, 0.65, 0.06, 0.60, 0.98, 0.19, 0.20, 0.44, 0.86},
    {0.56, 0.46, 0.95, 0.11, 0.12, 0.61, 0.04, 0.15, 0.02, 0.05, 0.29, 0.77, 0.49, 0.15, 0.65, 0.11, 0.80, 0.08, 0.40, 0.68, 0.72, 0.33},
    {0.72, 0.53, 0.10, 0.60, 0.47, 0.36, 0.79, 0.59, 0.73, 0.25, 0.10, 0.52, 0.45, 0.25, 0.41, 0.32, 0.58, 0.35, 0.42, 0.43, 0.91, 0.24},
    {0.90, 0.20, 0.50, 0.61, 0.86, 0.06, 0.28, 0.33, 0.66, 0.81, 0.16, 0.21, 0.96, 0.37, 0.20, 0.32, 0.35, 0.96, 0.01, 0.13, 0.44, 0.96},
    {0.97, 0.97, 0.62, 0.30, 0.78, 0.34, 0.51, 0.48, 0.73, 0.80, 0.71, 0.68, 0.60, 0.75, 0.09, 0.51, 0.79, 0.03, 0.54, 0.31, 0.36, 0.15},
    {0.31, 0.23, 0.77, 0.78, 0.56, 0.59, 0.97, 0.69, 0.56, 0.92, 0.02, 0.10, 0.34, 0.05, 0.64, 0.30, 0.54, 0.77, 0.58, 0.44, 0.52, 0.85},
    {0.20, 0.35, 0.05, 0.58, 0.79, 0.38, 0.39, 0.03, 0.80, 0.72, 0.07, 0.86, 0.13, 0.07, 0.55, 0.15, 0.15, 0.94, 0.68, 0.41, 0.06, 0.64},
    {0.01, 0.10, 0.89, 0.91, 0.80, 0.66, 0.89, 0.26, 0.27, 0.23, 0.80, 0.77, 0.06, 0.35, 0.51, 0.03, 0.17, 0.11, 0.34, 0.72, 0.45, 0.24},
    {0.51, 0.54, 0.68, 0.25, 0.53, 0.65, 0.83, 0.88, 0.83, 0.16, 0.79, 0.40, 0.84, 0.70, 0.65, 0.79, 0.73, 0.32, 0.31, 0.87, 0.76, 0.71},
    {0.32, 0.23, 0.84, 0.23, 0.63, 0.39, 0.44, 0.17, 0.56, 0.03, 0.80, 0.44, 0.62, 0.47, 0.86, 0.07, 0.34, 0.42, 0.75, 0.84, 0.35, 0.89},
    {0.60, 0.39, 0.70, 0.53, 0.68, 0.57, 0.62, 0.23, 0.83, 0.56, 0.67, 0.84, 0.28, 0.72, 0.51, 0.78, 0.48, 0.68, 0.98, 0.60, 0.37, 0.55}
};

void assert_create_lstm_cell(int inputCount, int nodesCount) {
    t_lstm_cell *lstm_cell = malloc(sizeof(t_lstm_cell));
    lstm_cell_init(lstm_cell, inputCount, nodesCount);
    lstm_cell_set_hidden_state_inputs(lstm_cell, hidden_state_inputs);
    lstm_cell_set_cell_state_inputs(lstm_cell, cell_state_inputs);
    lstm_cell_set_inputs(lstm_cell, inputs);
    lstm_cell_calculate_all_gates(lstm_cell);

    for (int node_index = 0; node_index < lstm_cell->state_vectors_size; node_index++) {
        assert((nodes_count_create_cell + inputs_count_create_cell + BIASES_COUNT) == lstm_cell->forget_gate->nodes[node_index].inputCount);
        for (int input_index = 0; input_index < lstm_cell->forget_gate->nodes[node_index].inputCount; input_index++) {
            lstm_cell->forget_gate->nodes[node_index].weights[input_index] = weights_for_all_nodes_forget_gate[node_index][input_index];
        }
    }
    for (int i = 0; i < lstm_cell->state_vectors_size; i++) {
        assert((nodes_count_create_cell + inputs_count_create_cell + BIASES_COUNT) == lstm_cell->input_gate->nodes[i].inputCount);
    }
    for (int i = 0; i < lstm_cell->state_vectors_size; i++) {
        assert((nodes_count_create_cell + inputs_count_create_cell + BIASES_COUNT) == lstm_cell->candidate_cell_state_gate->nodes[i].inputCount);
    }
    for (int i = 0; i < lstm_cell->state_vectors_size; i++) {
        assert((nodes_count_create_cell + inputs_count_create_cell + BIASES_COUNT) == lstm_cell->output_gate->nodes[i].inputCount);
    }

    for (int i = 0; i < lstm_cell->state_vectors_size; i++) {
        assert(hidden_state_inputs[i] == lstm_cell->inputs_to_layers[i]);
    }
    for (int i = lstm_cell->state_vectors_size; i < lstm_cell->inputs_count + lstm_cell->state_vectors_size; i++) {
        assert(inputs[i - lstm_cell->state_vectors_size] == lstm_cell->inputs_to_layers[i]);
    }
    lstm_cell_destroy(lstm_cell);
}

int main() {
    srand(time(NULL));
    assert_create_lstm_cell(inputs_count_create_cell, nodes_count_create_cell);
    return 0;
}
