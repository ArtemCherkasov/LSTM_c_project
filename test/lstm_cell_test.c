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
int inputs_count_create_cell_fp = 2;
int nodes_count_create_cell_fp = 2;

double hidden_state_inputs[] = {0.0801, -0.0812, -0.0879, 0.085, -0.0833, 0.082, -0.088, 0.084, -0.086, 0.083, -0.0891};
double cell_state_inputs[] = {0.101, 0.112, 0.179, 0.185, 0.133, 0.12, 0.18, 0.14, 0.16, 0.13, 0.191};
double inputs[] = {0.01, 0.12, -0.79, 0.5, 0.33, 0.2, -0.8, 0.4, 0.6, 0.3};
double hidden_state_inputs_fp[] = {0.0801, -0.0812};
double cell_state_inputs_fp[] = {0.101, 0.112};
double inputs_fp[] = {0.01, 0.12};
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

double weights_for_all_nodes_input_gate[11][22] =
{
    {0.96, 0.17, 0.34, 0.08, 0.40, 0.59, 0.86, 0.42, 0.29, 0.50, 0.32, 0.49, 0.50, 0.31, 0.04, 0.12, 0.57, 0.41, 0.35, 0.62, 0.42, 0.47},
    {0.90, 0.08, 0.72, 0.85, 0.10, 0.68, 0.12, 0.51, 0.67, 0.55, 0.00, 0.38, 0.13, 0.88, 0.10, 0.62, 0.01, 0.90, 0.14, 0.16, 0.66, 0.18},
    {0.05, 0.54, 0.87, 0.69, 0.18, 0.56, 0.81, 0.55, 0.66, 0.75, 0.33, 0.19, 0.44, 0.62, 0.09, 0.15, 0.93, 0.48, 0.36, 0.63, 0.84, 0.59},
    {0.56, 0.76, 0.08, 0.06, 0.49, 0.09, 0.63, 0.77, 0.01, 0.71, 0.33, 0.55, 0.17, 0.17, 0.47, 0.41, 0.07, 0.25, 0.62, 0.76, 0.08, 0.35},
    {0.53, 0.78, 0.14, 0.62, 0.14, 0.29, 0.31, 0.26, 0.73, 0.01, 0.55, 0.41, 0.98, 0.42, 0.02, 0.01, 0.60, 0.86, 0.60, 0.60, 0.58, 0.39},
    {0.38, 0.82, 0.97, 0.24, 0.33, 0.53, 0.34, 0.68, 0.51, 0.10, 0.49, 0.17, 0.05, 0.80, 0.59, 0.19, 0.18, 0.44, 0.78, 0.95, 0.03, 0.14},
    {0.76, 0.32, 0.73, 0.19, 0.53, 0.73, 0.55, 0.02, 0.94, 0.22, 0.67, 0.24, 0.00, 0.19, 0.93, 0.71, 0.64, 0.28, 0.12, 0.41, 0.21, 0.42},
    {0.95, 0.49, 0.65, 0.20, 0.73, 0.13, 0.77, 0.54, 0.42, 0.82, 0.04, 0.53, 0.35, 0.18, 0.78, 0.19, 0.89, 0.97, 0.52, 0.95, 0.43, 0.47},
    {0.14, 0.72, 0.06, 0.95, 0.43, 0.12, 0.79, 0.00, 0.70, 0.52, 0.48, 0.47, 0.40, 0.87, 0.46, 0.73, 0.96, 0.26, 0.76, 0.95, 0.53, 0.97},
    {0.83, 0.16, 0.97, 0.50, 0.94, 0.96, 0.89, 0.05, 0.60, 0.82, 0.76, 0.26, 0.25, 0.64, 0.84, 0.66, 0.91, 0.30, 0.17, 0.53, 0.14, 0.72},
    {0.27, 0.36, 0.21, 0.60, 0.04, 0.54, 0.59, 0.24, 0.08, 0.08, 0.04, 0.72, 0.54, 0.65, 0.02, 0.74, 0.18, 0.25, 0.99, 0.46, 0.47, 0.44}
};

double weights_for_all_nodes_candidate_cell_state_gate[11][22] =
{
    {0.82, 0.59, 0.86, 0.98, 0.92, 0.29, 0.21, 0.26, 0.02, 0.89, 0.58, 0.28, 0.11, 0.29, 0.34, 0.93, 0.97, 0.50, 0.92, 0.32, 0.26, 0.47},
    {0.80, 0.57, 0.78, 0.54, 0.46, 0.16, 0.34, 0.28, 0.55, 0.57, 0.69, 0.86, 0.48, 0.20, 0.27, 0.94, 0.38, 0.36, 0.69, 0.67, 0.72, 0.90},
    {0.04, 0.43, 0.31, 0.89, 0.76, 0.94, 0.47, 0.69, 0.96, 0.16, 0.63, 0.11, 0.11, 0.17, 0.97, 0.61, 0.36, 0.92, 0.29, 0.68, 0.43, 0.65},
    {0.41, 0.44, 0.67, 0.26, 0.85, 0.56, 0.34, 0.12, 0.08, 0.26, 0.64, 0.42, 0.47, 0.64, 0.89, 0.88, 0.72, 0.51, 0.96, 0.44, 0.78, 0.35},
    {0.15, 0.94, 0.87, 0.02, 0.45, 0.35, 0.94, 0.18, 0.26, 0.28, 0.31, 0.86, 0.17, 0.81, 0.50, 0.10, 0.03, 0.43, 0.04, 0.26, 0.27, 0.77},
    {0.79, 0.81, 0.92, 0.17, 0.38, 0.12, 0.34, 0.85, 0.95, 0.87, 0.54, 0.97, 0.19, 0.82, 0.37, 0.84, 0.29, 0.04, 0.24, 0.09, 0.87, 0.30},
    {0.33, 0.59, 0.86, 0.25, 0.23, 0.38, 0.54, 0.66, 0.80, 0.07, 0.94, 0.22, 0.50, 0.12, 0.65, 0.11, 0.09, 0.82, 0.90, 0.77, 0.87, 0.01},
    {0.65, 0.20, 0.65, 0.50, 0.53, 0.26, 0.57, 0.98, 0.71, 0.89, 0.48, 0.68, 0.95, 0.66, 0.69, 0.61, 0.61, 0.39, 0.73, 0.47, 0.94, 0.18},
    {0.56, 0.01, 0.31, 0.06, 0.01, 0.70, 0.62, 0.49, 0.65, 0.98, 0.53, 0.12, 0.31, 0.03, 0.54, 0.74, 0.69, 0.65, 0.32, 0.21, 0.71, 0.76},
    {0.20, 0.74, 0.93, 0.21, 0.41, 0.15, 0.02, 0.30, 0.97, 0.17, 0.88, 0.33, 0.79, 0.95, 0.01, 0.78, 0.84, 0.03, 0.32, 0.35, 0.69, 0.18},
    {0.72, 0.49, 0.20, 0.47, 0.93, 0.36, 0.39, 0.31, 0.30, 0.06, 0.56, 0.08, 0.94, 0.04, 0.88, 0.48, 0.51, 0.41, 0.00, 0.59, 0.42, 0.21}
};

double weights_for_all_nodes_output_gate[11][22] =
{
    {0.90, 0.67, 0.85, 0.70, 0.25, 0.37, 0.10, 0.45, 0.10, 0.38, 0.94, 0.94, 0.27, 0.16, 0.85, 0.49, 0.65, 0.43, 0.98, 0.91, 0.14, 0.65},
    {0.09, 0.35, 0.33, 0.11, 0.73, 0.70, 0.81, 0.65, 0.56, 0.17, 0.33, 0.82, 0.15, 0.26, 0.02, 0.07, 0.83, 0.73, 0.52, 0.42, 0.33, 0.66},
    {0.76, 0.93, 0.67, 0.23, 0.86, 0.96, 0.82, 0.75, 0.95, 0.22, 0.69, 0.40, 0.75, 0.70, 0.76, 0.06, 0.09, 0.19, 0.12, 0.30, 0.38, 0.18},
    {0.61, 0.83, 0.56, 0.32, 0.79, 0.29, 0.01, 0.27, 0.65, 0.69, 0.63, 0.46, 0.13, 0.88, 0.21, 0.63, 0.26, 0.55, 0.33, 0.39, 0.25, 0.27},
    {0.73, 0.01, 0.05, 0.48, 0.51, 0.02, 0.29, 0.84, 0.91, 0.81, 0.31, 0.19, 0.83, 0.24, 0.94, 0.64, 0.40, 0.16, 0.71, 0.48, 0.33, 0.79},
    {0.59, 0.32, 0.20, 0.21, 0.67, 0.85, 0.51, 0.23, 0.77, 0.29, 0.75, 0.84, 0.62, 0.73, 0.35, 0.02, 0.60, 0.75, 0.16, 0.80, 0.98, 0.29},
    {0.66, 0.43, 0.48, 0.40, 0.88, 0.17, 0.15, 0.18, 0.08, 0.21, 0.50, 0.35, 0.60, 0.61, 0.44, 0.96, 0.59, 0.91, 0.28, 0.38, 0.40, 0.48},
    {0.46, 0.02, 0.24, 0.04, 0.14, 0.98, 0.83, 0.71, 0.25, 0.61, 0.63, 0.54, 0.99, 0.79, 0.91, 0.80, 0.19, 0.32, 0.33, 0.66, 0.40, 0.05},
    {0.14, 0.31, 0.46, 0.42, 0.01, 0.07, 0.57, 0.80, 0.01, 0.43, 0.30, 0.50, 0.40, 0.54, 0.22, 0.07, 0.38, 0.96, 0.92, 0.44, 0.59, 0.18},
    {0.39, 0.63, 0.74, 0.66, 0.55, 0.93, 0.46, 0.34, 0.12, 0.95, 0.90, 0.32, 0.76, 0.56, 0.41, 0.27, 0.37, 0.28, 0.11, 0.23, 0.32, 0.90},
    {0.04, 0.68, 0.20, 0.99, 0.52, 0.70, 0.49, 0.67, 0.48, 0.86, 0.27, 0.59, 0.33, 0.45, 0.73, 0.99, 0.05, 0.30, 0.24, 0.61, 0.61, 0.40}
};

double weights_for_all_nodes_forget_gate_fp[2][5] =
{
    {0.65, 0.60, 0.76, 0.49, 0.21},
    {0.56, 0.46, 0.95, 0.11, 0.12}
};

double weights_for_all_nodes_input_gate_fp[2][5] =
{
    {0.96, 0.17, 0.34, 0.08, 0.40},
    {0.90, 0.08, 0.72, 0.85, 0.10}
};

double weights_for_all_nodes_candidate_cell_state_gate_fp[2][5] =
{
    {0.82, 0.59, 0.86, 0.98, 0.92},
    {0.80, 0.57, 0.78, 0.54, 0.46}
};

double weights_for_all_nodes_output_gate_fp[2][5] =
{
    {0.90, 0.67, 0.85, 0.70, 0.25},
    {0.09, 0.35, 0.33, 0.11, 0.73}
};

double forget_gate_output[] =
{
    0.62345804545385696827,
    0.81432232138059934545,
    0.71577811391083701054,
    0.62314670527621418561,
    0.58958256653781815348,
    0.80546724773313294055,
    0.66778435314346329399,
    0.67692806405466476871,
    0.81023271450781919611,
    0.81493194735437202159,
    0.69439997749447224251
};

double cell_state_output_fp[] =
{
    0.54306992827602940110,
    0.34479679036474636389
};

double hidden_state_output_fp[] =
{
    0.29177889909371879096,
    0.22352544083548625542
};

void assert_create_lstm_cell(int inputCount, int nodesCount) {
    t_lstm_cell *lstm_cell = malloc(sizeof(t_lstm_cell));
    lstm_cell_init(lstm_cell, inputCount, nodesCount);
    lstm_cell_set_hidden_state_inputs(lstm_cell, hidden_state_inputs);
    lstm_cell_set_cell_state_inputs(lstm_cell, cell_state_inputs);
    lstm_cell_set_inputs(lstm_cell, inputs);

    for (int node_index = 0; node_index < lstm_cell->state_vectors_size; node_index++) {
        assert((nodes_count_create_cell + inputs_count_create_cell + BIASES_COUNT) == lstm_cell->forget_gate->nodes[node_index].inputCount);
        for (int input_index = 0; input_index < lstm_cell->inputs_to_layers_count_without_biases + BIASES_COUNT; input_index++) {
            lstm_cell->forget_gate->nodes[node_index].weights[input_index] = weights_for_all_nodes_forget_gate[node_index][input_index];
        }
    }
    for (int node_index = 0; node_index < lstm_cell->state_vectors_size; node_index++) {
        assert((nodes_count_create_cell + inputs_count_create_cell + BIASES_COUNT) == lstm_cell->input_gate->nodes[node_index].inputCount);
        for (int input_index = 0; input_index < lstm_cell->inputs_to_layers_count_without_biases + BIASES_COUNT; input_index++) {
            lstm_cell->input_gate->nodes[node_index].weights[input_index] = weights_for_all_nodes_input_gate[node_index][input_index];
        }
    }
    for (int node_index = 0; node_index < lstm_cell->state_vectors_size; node_index++) {
        assert((nodes_count_create_cell + inputs_count_create_cell + BIASES_COUNT) == lstm_cell->candidate_cell_state_gate->nodes[node_index].inputCount);
        for (int input_index = 0; input_index < lstm_cell->inputs_to_layers_count_without_biases + BIASES_COUNT; input_index++) {
            lstm_cell->candidate_cell_state_gate->nodes[node_index].weights[input_index] = weights_for_all_nodes_candidate_cell_state_gate[node_index][input_index];
        }
    }
    for (int node_index = 0; node_index < lstm_cell->state_vectors_size; node_index++) {
        assert((nodes_count_create_cell + inputs_count_create_cell + BIASES_COUNT) == lstm_cell->output_gate->nodes[node_index].inputCount);
        for (int input_index = 0; input_index < lstm_cell->inputs_to_layers_count_without_biases + BIASES_COUNT; input_index++) {
            lstm_cell->output_gate->nodes[node_index].weights[input_index] = weights_for_all_nodes_output_gate[node_index][input_index];
        }
    }

    lstm_cell_calculate_all_gates(lstm_cell);

    for (int i = 0; i < lstm_cell->state_vectors_size; i++) {
        assert(hidden_state_inputs[i] == lstm_cell->inputs_to_layers[i]);
    }
    for (int i = lstm_cell->state_vectors_size; i < lstm_cell->inputs_to_layers_count_without_biases; i++) {
        assert(inputs[i - lstm_cell->state_vectors_size] == lstm_cell->inputs_to_layers[i]);
    }
    for (int node_index = 0; node_index < lstm_cell->state_vectors_size; node_index++) {
        assert(forget_gate_output[node_index] == lstm_cell->forget_gate->output[node_index]);
    }
    lstm_cell_destroy(lstm_cell);
}

void assert_create_lstm_cell_forward_propagation(int inputCount, int nodesCount) {
    t_lstm_cell *lstm_cell = malloc(sizeof(t_lstm_cell));
    lstm_cell_init(lstm_cell, inputCount, nodesCount);
    lstm_cell_set_hidden_state_inputs(lstm_cell, hidden_state_inputs_fp);
    lstm_cell_set_cell_state_inputs(lstm_cell, cell_state_inputs_fp);
    lstm_cell_set_inputs(lstm_cell, inputs_fp);
    for (int node_index = 0; node_index < lstm_cell->state_vectors_size; node_index++) {
        for (int input_index = 0; input_index < lstm_cell->inputs_to_layers_count_without_biases + BIASES_COUNT; input_index++) {
            lstm_cell->forget_gate->nodes[node_index].weights[input_index] = weights_for_all_nodes_forget_gate_fp[node_index][input_index];
        }
    }
    for (int node_index = 0; node_index < lstm_cell->state_vectors_size; node_index++) {
        for (int input_index = 0; input_index < lstm_cell->inputs_to_layers_count_without_biases + BIASES_COUNT; input_index++) {
            lstm_cell->input_gate->nodes[node_index].weights[input_index] = weights_for_all_nodes_input_gate_fp[node_index][input_index];
        }
    }
    for (int node_index = 0; node_index < lstm_cell->state_vectors_size; node_index++) {
        for (int input_index = 0; input_index < lstm_cell->inputs_to_layers_count_without_biases + BIASES_COUNT; input_index++) {
            lstm_cell->candidate_cell_state_gate->nodes[node_index].weights[input_index] = weights_for_all_nodes_candidate_cell_state_gate_fp[node_index][input_index];
        }
    }
    for (int node_index = 0; node_index < lstm_cell->state_vectors_size; node_index++) {
        for (int input_index = 0; input_index < lstm_cell->inputs_to_layers_count_without_biases + BIASES_COUNT; input_index++) {
            lstm_cell->output_gate->nodes[node_index].weights[input_index] = weights_for_all_nodes_output_gate_fp[node_index][input_index];
        }
    }
    lstm_cell_forward_propagation(lstm_cell);
    for (int node_index = 0; node_index < lstm_cell->state_vectors_size; node_index++) {
        assert(cell_state_output_fp[node_index] == lstm_cell->cell_state[node_index]);
        assert(hidden_state_output_fp[node_index] == lstm_cell->hidden_state[node_index]);
    }
}

int main() {
    srand(time(NULL));
    assert_create_lstm_cell(inputs_count_create_cell, nodes_count_create_cell);
    assert_create_lstm_cell_forward_propagation(inputs_count_create_cell_fp, nodes_count_create_cell_fp);
    return 0;
}
