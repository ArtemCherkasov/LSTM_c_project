//
// Created by User on 08.06.2026.
//

#include "../src/nn/lstm/lstm_cell/lstm_cell.h"

#include <stdlib.h>
#include <assert.h>
#include <math.h>
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
int forecast_cells_count_nn = 3;
int inputs_count_create_cells_nn = 2;
int nodes_count_create_cells_nn = 2;

double hidden_state_inputs_nn[] = {0.0801, -0.0812};
double cell_state_inputs_nn[] = {0.101, 0.112};
double inputs_nn_0[] = {0.01, 0.12};
double inputs_nn_1[] = {0.03, 0.04};
double expected_vector[2][2] = {
    {0.10736, 0.10242},
    {0.10554, 0.10903}
};
double expected_vector_with_empty_input[5][2] = {
    {0.10736, 0.10242},
    {0.10334, 0.12668},
    {0.10511, 0.11099},
    {0.10599, 0.14122},
    {0.10888, 0.10089}
};
double forecast_vector[2][2] = {
    {0.10736464436284548496, 0.10240871204587850829},
    {0.10555463954661467585, 0.10900233271451564843}
};

double forecast_vector_hidden_state_with_empty_input_0[5][2] = {
    {0.256071179266923, 0.178266105531015},
    {0.461001464468132, 0.354273300148213},
    {0.705756156571722, 0.500432377036408},
    {0.812611354835564, 0.594316699231035},
    {0.858352717064926, 0.631655697418286}
};

double forecast_vector_hidden_state_with_empty_input_1[5][2] = {
    {0.316968948788424, 0.257645409824590},
    {0.635209812189013, 0.540098295508523},
    {0.839389691541576, 0.714762599252062},
    {0.914319313199073, 0.781285471836479},
    {0.936237183234012, 0.802250007328005}
};

double forecast_vector_hidden_state_after_learn_with_empty_input[5][2] = {
    {0.10736064654408965235, 0.10241846487744078742},
    {0.10334019737147279172, 0.12667884733672027386},
    {0.10511143924135052385, 0.11098836974272216493},
    {0.10599147248300723467, 0.14121787709232766495},
    {0.10887942830915124659, 0.10088939420627214549}
};

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

double hidden_state_second_layer_output_nn[] =
{
    0.37060546865140853168,
    0.32202646443203358784
};

void assert_lstm_network_pass_inputs(int cellsCount, int inputCount, int nodesCount) {
    t_lstm_neural_network *lstm_network = malloc(sizeof(t_lstm_neural_network));
    lstm_neural_network_init(lstm_network, cellsCount, inputCount, nodesCount);
    lstm_neural_network_set_input(lstm_network, inputs_pass_assert);
    int inputs_pass_assert_index = 0;
    for (int cell_index = 0; cell_index < lstm_network->cells_count_full; cell_index++) {
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

    for (int cell_index = 0; cell_index < lstm_network->cells_count_full; cell_index++) {
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

    lstm_cell_set_hidden_state_inputs(&lstm_network->lstm_cells[0], hidden_state_inputs_nn);
    lstm_cell_set_cell_state_inputs(&lstm_network->lstm_cells[0], cell_state_inputs_nn);
    lstm_cell_set_inputs(&lstm_network->lstm_cells[0], inputs_nn_0);
    lstm_cell_set_inputs(&lstm_network->lstm_cells[1], inputs_nn_1);
    lstm_neural_network_forward_propagation(lstm_network);

    for (int node_index = 0; node_index < lstm_network->lstm_cells[0].state_vectors_size; node_index++) {
        assert(cell_state_output_nn[node_index] == lstm_network->lstm_cells[0].cell_state[node_index]);
        assert(hidden_state_output_nn[node_index] == lstm_network->lstm_cells[0].hidden_state[node_index]);
    }

    lstm_neural_network_destroy(lstm_network);
    free(lstm_network);
}

void assert_double_lstm_network_forward_propagation(int cellsCount, int inputCount, int nodesCount) {
    t_lstm_neural_network *lstm_network = malloc(sizeof(t_lstm_neural_network));
    lstm_neural_network_init(lstm_network, cellsCount, inputCount, nodesCount);
    lstm_network->index = 0;
    lstm_network->learning_rate = 0.15;
    lstm_network->next = malloc(sizeof(t_lstm_neural_network));
    lstm_neural_network_init(lstm_network->next, cellsCount, inputCount, nodesCount);
    lstm_network->next->index = 1;
    lstm_network->next->learning_rate = 0.15;

    for (int cell_index = 0; cell_index < lstm_network->cells_count_full; cell_index++) {
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

    for (int cell_index = 0; cell_index < lstm_network->next->cells_count_full; cell_index++) {
        for (int node_index = 0; node_index < lstm_network->next->lstm_cells[cell_index].state_vectors_size; node_index++) {
            for (int input_index = 0; input_index < lstm_network->next->lstm_cells[cell_index].inputs_to_layers_count_without_biases + BIASES_COUNT; input_index++) {
                lstm_network->next->lstm_cells[cell_index].forget_gate->nodes[node_index].weights[input_index] = weights_for_all_nodes_forget_gate_nn[node_index][input_index];
            }
        }
        for (int node_index = 0; node_index < lstm_network->next->lstm_cells[cell_index].state_vectors_size; node_index++) {
            for (int input_index = 0; input_index < lstm_network->next->lstm_cells[cell_index].inputs_to_layers_count_without_biases + BIASES_COUNT; input_index++) {
                lstm_network->next->lstm_cells[cell_index].input_gate->nodes[node_index].weights[input_index] = weights_for_all_nodes_input_gate_nn[node_index][input_index];
            }
        }
        for (int node_index = 0; node_index < lstm_network->next->lstm_cells[cell_index].state_vectors_size; node_index++) {
            for (int input_index = 0; input_index < lstm_network->next->lstm_cells[cell_index].inputs_to_layers_count_without_biases + BIASES_COUNT; input_index++) {
                lstm_network->next->lstm_cells[cell_index].candidate_cell_state_gate->nodes[node_index].weights[input_index] = weights_for_all_nodes_candidate_cell_state_gate_nn[node_index][input_index];
            }
        }
        for (int node_index = 0; node_index < lstm_network->next->lstm_cells[cell_index].state_vectors_size; node_index++) {
            for (int input_index = 0; input_index < lstm_network->next->lstm_cells[cell_index].inputs_to_layers_count_without_biases + BIASES_COUNT; input_index++) {
                lstm_network->next->lstm_cells[cell_index].output_gate->nodes[node_index].weights[input_index] = weights_for_all_nodes_output_gate_nn[node_index][input_index];
            }
        }
    }

    lstm_cell_set_hidden_state_inputs(&lstm_network->lstm_cells[0], hidden_state_inputs_nn);
    lstm_cell_set_cell_state_inputs(&lstm_network->lstm_cells[0], cell_state_inputs_nn);
    lstm_cell_set_hidden_state_inputs(&lstm_network->next->lstm_cells[0], hidden_state_inputs_nn);
    lstm_cell_set_cell_state_inputs(&lstm_network->next->lstm_cells[0], cell_state_inputs_nn);
    lstm_cell_set_inputs(&lstm_network->lstm_cells[0], inputs_nn_0);
    lstm_cell_set_inputs(&lstm_network->lstm_cells[1], inputs_nn_1);
    lstm_neural_network_forward_propagation(lstm_network);

    for (int node_index = 0; node_index < lstm_network->lstm_cells[0].state_vectors_size; node_index++) {
        assert(hidden_state_output_nn[node_index] == lstm_network->next->lstm_cells[0].inputs[node_index]);
        assert(hidden_state_second_layer_output_nn[node_index] == lstm_network->next->lstm_cells[0].hidden_state[node_index]);
    }

    lstm_cell_set_expected_vector(&lstm_network->next->lstm_cells[0], expected_vector[0]);
    lstm_cell_set_expected_vector(&lstm_network->next->lstm_cells[1], expected_vector[1]);
    for (int index = 0; index < 50; ++index) {
        lstm_neural_network_learning_step(lstm_network);
    }

    lstm_network->learning_rate = 0.001;
    lstm_network->next->learning_rate = 0.001;

    for (int index = 0; index < 300; ++index) {
        lstm_neural_network_learning_step(lstm_network);
    }

    lstm_network->learning_rate = 0.00005;
    lstm_network->next->learning_rate = 0.00005;
    for (int index = 0; index < 200; ++index) {
        lstm_neural_network_learning_step(lstm_network);
    }

    for (int cell_index = 0; cell_index < lstm_network->next->cells_count_full; cell_index++) {
        assert(forecast_vector[cell_index][0] == lstm_network->next->lstm_cells[cell_index].hidden_state[0]);
        assert(forecast_vector[cell_index][1] == lstm_network->next->lstm_cells[cell_index].hidden_state[1]);
    }
    lstm_neural_network_full_mean_squared_error_calculation(lstm_network);
    lstm_neural_network_destroy(lstm_network->next);
    lstm_neural_network_destroy(lstm_network);
    free(lstm_network->next);
    free(lstm_network);
}

void assert_double_lstm_network_forward_propagation_with_forecast_vector(int cellsCount, int inputCount, int nodesCount, int forecastCellsCount) {
    t_lstm_neural_network *lstm_network = malloc(sizeof(t_lstm_neural_network));
    lstm_neural_network_init_with_empty_input_vector(lstm_network, cellsCount, inputCount, nodesCount, forecastCellsCount);
    lstm_network->index = 0;
    lstm_network->learning_rate = 0.15;
    lstm_network->next = malloc(sizeof(t_lstm_neural_network));
    lstm_neural_network_init(lstm_network->next, cellsCount + forecastCellsCount, inputCount, nodesCount);
    lstm_network->next->index = 1;
    lstm_network->next->learning_rate = 0.15;
    lstm_cell_set_inputs(&lstm_network->lstm_cells[0], inputs_nn_0);
    lstm_cell_set_inputs(&lstm_network->lstm_cells[1], inputs_nn_1);

    lstm_cell_set_expected_vector(&lstm_network->next->lstm_cells[0], expected_vector_with_empty_input[0]);
    lstm_cell_set_expected_vector(&lstm_network->next->lstm_cells[1], expected_vector_with_empty_input[1]);
    lstm_cell_set_expected_vector(&lstm_network->next->lstm_cells[2], expected_vector_with_empty_input[2]);
    lstm_cell_set_expected_vector(&lstm_network->next->lstm_cells[3], expected_vector_with_empty_input[3]);
    lstm_cell_set_expected_vector(&lstm_network->next->lstm_cells[4], expected_vector_with_empty_input[4]);

    for (int cell_index = 0; cell_index < lstm_network->cells_count_full; cell_index++) {
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

    for (int cell_index = 0; cell_index < lstm_network->next->cells_count_full; cell_index++) {
        for (int node_index = 0; node_index < lstm_network->next->lstm_cells[cell_index].state_vectors_size; node_index++) {
            for (int input_index = 0; input_index < lstm_network->next->lstm_cells[cell_index].inputs_to_layers_count_without_biases + BIASES_COUNT; input_index++) {
                lstm_network->next->lstm_cells[cell_index].forget_gate->nodes[node_index].weights[input_index] = weights_for_all_nodes_forget_gate_nn[node_index][input_index];
            }
        }
        for (int node_index = 0; node_index < lstm_network->next->lstm_cells[cell_index].state_vectors_size; node_index++) {
            for (int input_index = 0; input_index < lstm_network->next->lstm_cells[cell_index].inputs_to_layers_count_without_biases + BIASES_COUNT; input_index++) {
                lstm_network->next->lstm_cells[cell_index].input_gate->nodes[node_index].weights[input_index] = weights_for_all_nodes_input_gate_nn[node_index][input_index];
            }
        }
        for (int node_index = 0; node_index < lstm_network->next->lstm_cells[cell_index].state_vectors_size; node_index++) {
            for (int input_index = 0; input_index < lstm_network->next->lstm_cells[cell_index].inputs_to_layers_count_without_biases + BIASES_COUNT; input_index++) {
                lstm_network->next->lstm_cells[cell_index].candidate_cell_state_gate->nodes[node_index].weights[input_index] = weights_for_all_nodes_candidate_cell_state_gate_nn[node_index][input_index];
            }
        }
        for (int node_index = 0; node_index < lstm_network->next->lstm_cells[cell_index].state_vectors_size; node_index++) {
            for (int input_index = 0; input_index < lstm_network->next->lstm_cells[cell_index].inputs_to_layers_count_without_biases + BIASES_COUNT; input_index++) {
                lstm_network->next->lstm_cells[cell_index].output_gate->nodes[node_index].weights[input_index] = weights_for_all_nodes_output_gate_nn[node_index][input_index];
            }
        }
    }
    lstm_neural_network_forward_propagation(lstm_network);
    assert(fabs(forecast_vector_hidden_state_with_empty_input_0[0][0] - lstm_network->lstm_cells[0].hidden_state[0]) < EPS);

    for (int index = 0; index < 100; ++index) {
        lstm_neural_network_learning_step(lstm_network);
    }

    lstm_network->learning_rate = 0.015;
    lstm_network->next->learning_rate = 0.015;
    for (int index = 0; index < 100; ++index) {
        lstm_neural_network_learning_step(lstm_network);
    }
    lstm_network->learning_rate = 0.0015;
    lstm_network->next->learning_rate = 0.0015;
    for (int index = 0; index < 20; ++index) {
        lstm_neural_network_learning_step(lstm_network);
    }
    lstm_network->learning_rate = 0.00015;
    lstm_network->next->learning_rate = 0.00015;
    for (int index = 0; index < 20; ++index) {
        lstm_neural_network_learning_step(lstm_network);
    }
    lstm_network->learning_rate = 0.000015;
    lstm_network->next->learning_rate = 0.000015;
    for (int index = 0; index < 20; ++index) {
        lstm_neural_network_learning_step(lstm_network);
    }
    lstm_network->learning_rate = 0.000005;
    lstm_network->next->learning_rate = 0.000005;
    for (int index = 0; index < 10; ++index) {
        lstm_neural_network_learning_step(lstm_network);
    }
    for (int cell_index = 0; cell_index < lstm_network->next->cells_count_full; cell_index++) {
        assert(fabs(forecast_vector_hidden_state_after_learn_with_empty_input[cell_index][0] - lstm_network->next->lstm_cells[cell_index].hidden_state[0]) < EPS);
        assert(fabs(forecast_vector_hidden_state_after_learn_with_empty_input[cell_index][1] - lstm_network->next->lstm_cells[cell_index].hidden_state[1]) < EPS);
    }
}

void assert_double_lstm_network_bptt(int cellsCount, int inputCount, int nodesCount, int forecastCellsCount) {
    t_lstm_neural_network *lstm_network = malloc(sizeof(t_lstm_neural_network));
    lstm_neural_network_init_with_empty_input_vector(lstm_network, cellsCount, inputCount, nodesCount, forecastCellsCount);
    lstm_network->index = 0;
    lstm_network->learning_rate = 0.15;
    lstm_network->next = malloc(sizeof(t_lstm_neural_network));
    lstm_neural_network_init(lstm_network->next, cellsCount + forecastCellsCount, inputCount, nodesCount);
    lstm_network->next->index = 1;
    lstm_network->next->learning_rate = 0.15;
    lstm_cell_set_inputs(&lstm_network->lstm_cells[0], inputs_nn_0);
    lstm_cell_set_inputs(&lstm_network->lstm_cells[1], inputs_nn_1);

    for (int cell_index = 0; cell_index < lstm_network->cells_count_full; cell_index++) {
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

    for (int cell_index = 0; cell_index < lstm_network->next->cells_count_full; cell_index++) {
        for (int node_index = 0; node_index < lstm_network->next->lstm_cells[cell_index].state_vectors_size; node_index++) {
            for (int input_index = 0; input_index < lstm_network->next->lstm_cells[cell_index].inputs_to_layers_count_without_biases + BIASES_COUNT; input_index++) {
                lstm_network->next->lstm_cells[cell_index].forget_gate->nodes[node_index].weights[input_index] = weights_for_all_nodes_forget_gate_nn[node_index][input_index];
            }
        }
        for (int node_index = 0; node_index < lstm_network->next->lstm_cells[cell_index].state_vectors_size; node_index++) {
            for (int input_index = 0; input_index < lstm_network->next->lstm_cells[cell_index].inputs_to_layers_count_without_biases + BIASES_COUNT; input_index++) {
                lstm_network->next->lstm_cells[cell_index].input_gate->nodes[node_index].weights[input_index] = weights_for_all_nodes_input_gate_nn[node_index][input_index];
            }
        }
        for (int node_index = 0; node_index < lstm_network->next->lstm_cells[cell_index].state_vectors_size; node_index++) {
            for (int input_index = 0; input_index < lstm_network->next->lstm_cells[cell_index].inputs_to_layers_count_without_biases + BIASES_COUNT; input_index++) {
                lstm_network->next->lstm_cells[cell_index].candidate_cell_state_gate->nodes[node_index].weights[input_index] = weights_for_all_nodes_candidate_cell_state_gate_nn[node_index][input_index];
            }
        }
        for (int node_index = 0; node_index < lstm_network->next->lstm_cells[cell_index].state_vectors_size; node_index++) {
            for (int input_index = 0; input_index < lstm_network->next->lstm_cells[cell_index].inputs_to_layers_count_without_biases + BIASES_COUNT; input_index++) {
                lstm_network->next->lstm_cells[cell_index].output_gate->nodes[node_index].weights[input_index] = weights_for_all_nodes_output_gate_nn[node_index][input_index];
            }
        }
    }

    lstm_neural_network_forward_propagation(lstm_network);
    printf("hidden %0.15f  %0.15f\n", lstm_network->next->lstm_cells[1].hidden_state[0], lstm_network->next->lstm_cells[1].hidden_state[1]);
    //TODO
}

int main() {
    srand(time(NULL));
    assert_lstm_network_pass_inputs(cells_count_pass_inputs, inputs_count_create_cell_pass_inputs, nodes_count_create_cell_pass_inputs);
    assert_lstm_network_forward_propagation(cells_count_nn, inputs_count_create_cells_nn, nodes_count_create_cells_nn);
    assert_double_lstm_network_forward_propagation(cells_count_nn, inputs_count_create_cells_nn, nodes_count_create_cells_nn);
    assert_double_lstm_network_forward_propagation_with_forecast_vector(cells_count_nn, inputs_count_create_cells_nn, nodes_count_create_cells_nn, forecast_cells_count_nn);
    assert_double_lstm_network_bptt(cells_count_nn, inputs_count_create_cells_nn, nodes_count_create_cells_nn, forecast_cells_count_nn);
    return 0;
}
