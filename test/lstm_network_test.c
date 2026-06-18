//
// Created by User on 08.06.2026.
//

#include "../src/nn/lstm/lstm_cell/lstm_cell.h"

#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

#include "../src/nn/common/layer/layer.h"

#include <time.h>

#include "../src/nn/lstm/lstm_neural_network/lstm_neural_network.h"

int cells_count_pass_inputs = 20;
int inputs_count_create_cell_pass_inputs = 2;
int nodes_count_create_cell_pass_inputs = 2;
double inputs_pass_assert[] = {
    0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
    0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20,
    0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30,
    0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40
};
int cells_count_nn = 2;
int inputs_count_create_cell_nn = 2;
int nodes_count_create_cell_nn = 2;

double hidden_state_inputs_nn[] = {0.0801, -0.0812};
double cell_state_inputs_nn[] = {0.101, 0.112};
double inputs_nn[] = {0.01, 0.12};

double weights_for_all_nodes_forget_gate_nn[2][5] =
{
    {0.65, 0.60, 0.76, 0.49, 0.21},
    {0.56, 0.46, 0.95, 0.11, 0.12}
};

double weights_for_all_nodes_input_gate_nn[2][5] =
{
    {0.96, 0.17, 0.34, 0.08, 0.40},
    {0.90, 0.08, 0.72, 0.85, 0.10}
};

double weights_for_all_nodes_candidate_cell_state_gate_nn[2][5] =
{
    {0.82, 0.59, 0.86, 0.98, 0.92},
    {0.80, 0.57, 0.78, 0.54, 0.46}
};

double weights_for_all_nodes_output_gate_nn[2][5] =
{
    {0.90, 0.67, 0.85, 0.70, 0.25},
    {0.09, 0.35, 0.33, 0.11, 0.73}
};

double cell_state_output_nn[] =
{
    0.54306992827602940110,
    0.34479679036474636389
};

double hidden_state_output_nn[] =
{
    0.29177889909371879096,
    0.22352544083548625542
};

void assert_lstm_network_pass_inputs(int cellsCount, int inputCount, int nodesCount) {
    t_lstm_neural_network *lstm_network = malloc(sizeof(t_lstm_neural_network));
    lstm_neural_network_init(lstm_network, cellsCount, inputCount, nodesCount);
    lstm_neural_network_set_input(lstm_network, inputs_pass_assert);
    int inputs_pass_assert_index = 0;
    for (int cell_index = 0; cell_index < lstm_network->cells_count; cell_index++) {
        for (int input_index = 0; input_index < lstm_network->inputs_count_per_cell; input_index++) {
            assert(inputs_pass_assert[inputs_pass_assert_index] == lstm_network->lstm_cells[cell_index].inputs[input_index]);
            inputs_pass_assert_index++;
        }
    }
    lstm_neural_network_destroy(lstm_network);
    free(lstm_network);
}

void assert_lstm_network_forward_propagation(int cellsCount, int inputCount, int nodesCount) {
    t_lstm_neural_network *lstm_network = malloc(sizeof(t_lstm_neural_network));
    lstm_neural_network_init(lstm_network, cellsCount, inputCount, nodesCount);

    for (int cell_index = 0; cell_index < lstm_network->cells_count; cell_index++) {
        for (int node_index = 0; node_index < lstm_network->lstm_cells[cell_index].state_vectors_size; node_index++) {
            for (int input_index = 0; input_index < lstm_network->lstm_cells[cell_index].inputs_to_layers_count_without_biases + BIASES_COUNT; input_index++) {
                lstm_network->lstm_cells[cell_index].forget_gate->nodes[node_index].weights[input_index] = weights_for_all_nodes_forget_gate_nn[node_index][input_index];
            }
        }
        for (int node_index = 0; node_index < lstm_network->lstm_cells[cell_index].state_vectors_size; node_index++) {
            for (int input_index = 0; input_index < lstm_network->lstm_cells[cell_index].inputs_to_layers_count_without_biases + BIASES_COUNT; input_index++) {
                lstm_network->lstm_cells[cell_index].input_gate->nodes[node_index].weights[input_index] = weights_for_all_nodes_input_gate_nn[node_index][input_index];
            }
        }
        for (int node_index = 0; node_index < lstm_network->lstm_cells[cell_index].state_vectors_size; node_index++) {
            for (int input_index = 0; input_index < lstm_network->lstm_cells[cell_index].inputs_to_layers_count_without_biases + BIASES_COUNT; input_index++) {
                lstm_network->lstm_cells[cell_index].candidate_cell_state_gate->nodes[node_index].weights[input_index] = weights_for_all_nodes_candidate_cell_state_gate_nn[node_index][input_index];
            }
        }
        for (int node_index = 0; node_index < lstm_network->lstm_cells[cell_index].state_vectors_size; node_index++) {
            for (int input_index = 0; input_index < lstm_network->lstm_cells[cell_index].inputs_to_layers_count_without_biases + BIASES_COUNT; input_index++) {
                lstm_network->lstm_cells[cell_index].output_gate->nodes[node_index].weights[input_index] = weights_for_all_nodes_output_gate_nn[node_index][input_index];
            }
        }
    }

    lstm_neural_network_set_input(lstm_network, inputs_pass_assert);
    lstm_cell_set_hidden_state_inputs(&lstm_network->lstm_cells[0], hidden_state_inputs_nn);
    lstm_cell_set_cell_state_inputs(&lstm_network->lstm_cells[0], cell_state_inputs_nn);
    lstm_cell_set_inputs(&lstm_network->lstm_cells[0], inputs_nn);
    lstm_neural_network_forward_propagation(lstm_network);

    for (int node_index = 0; node_index < lstm_network->lstm_cells[0].state_vectors_size; node_index++) {
        assert(cell_state_output_nn[node_index] == lstm_network->lstm_cells[0].cell_state[node_index]);
        assert(hidden_state_output_nn[node_index] == lstm_network->lstm_cells[0].hidden_state[node_index]);
    }

    lstm_neural_network_destroy(lstm_network);
    free(lstm_network);
}

int main() {
    srand(time(NULL));
    assert_lstm_network_pass_inputs(cells_count_pass_inputs, inputs_count_create_cell_pass_inputs, nodes_count_create_cell_pass_inputs);
    assert_lstm_network_forward_propagation(cells_count_nn, inputs_count_create_cell_nn, nodes_count_create_cell_nn);
    return 0;
}
