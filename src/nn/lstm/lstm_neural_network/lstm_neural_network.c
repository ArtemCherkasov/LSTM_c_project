//
// Created by User on 18.06.2026.
//

#include "lstm_neural_network.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void lstm_neural_network_init(t_lstm_neural_network *lstm_network, int cells_count, int inputs_count_per_cell, int nodes_count_per_cell) {
	lstm_network->cells_count = cells_count;
	lstm_network->inputs_count_per_cell = inputs_count_per_cell;
	lstm_network->nodes_count_per_cell = nodes_count_per_cell;
	lstm_network->inputs_count_per_network = inputs_count_per_cell * cells_count;
	lstm_network->lstm_cells = malloc(sizeof(t_lstm_cell) * cells_count);
	for (int cell_index = 0; cell_index < lstm_network->cells_count; cell_index++) {
		lstm_cell_init(&lstm_network->lstm_cells[cell_index], lstm_network->inputs_count_per_cell, lstm_network->nodes_count_per_cell);
	}
}

void lstm_neural_network_set_input(t_lstm_neural_network *lstm_network, double *inputs) {
	for (int cell_index = 0; cell_index < lstm_network->cells_count; cell_index++) {
		memcpy(lstm_network->lstm_cells[cell_index].inputs, &inputs[lstm_network->inputs_count_per_cell * cell_index], sizeof(double) * lstm_network->inputs_count_per_cell);
	}
}

void lstm_neural_network_forward_propagation(t_lstm_neural_network *lstm_network) {
	for (int cell_index = 0; cell_index < lstm_network->cells_count; cell_index++) {
		lstm_cell_forward_propagation(&lstm_network->lstm_cells[cell_index]);
		if (cell_index < lstm_network->cells_count - 1) {
			lstm_cell_set_cell_state_inputs(&lstm_network->lstm_cells[cell_index + 1], lstm_network->lstm_cells[cell_index].cell_state);
			lstm_cell_set_hidden_state_inputs(&lstm_network->lstm_cells[cell_index + 1], lstm_network->lstm_cells[cell_index].hidden_state);
		}
	}
}

void lstm_neural_network_forward_propagation_from_cell_index(t_lstm_neural_network *lstm_network, int from_cell_index) {
	for (int cell_index = from_cell_index; cell_index < lstm_network->cells_count; cell_index++) {
		lstm_cell_forward_propagation(&lstm_network->lstm_cells[cell_index]);
		if (cell_index < lstm_network->cells_count - 1) {
			lstm_cell_set_cell_state_inputs(&lstm_network->lstm_cells[cell_index + 1], lstm_network->lstm_cells[cell_index].cell_state);
			lstm_cell_set_hidden_state_inputs(&lstm_network->lstm_cells[cell_index + 1], lstm_network->lstm_cells[cell_index].hidden_state);
		}
	}
}

void lstm_neural_network_full_mean_squared_error_calculation(t_lstm_neural_network *lstm_network) {
	lstm_network->full_mean_squared_error = 0;
	for (int cell_index = 0; cell_index < lstm_network->cells_count; cell_index++) {
		for (int node_index = 0; node_index < lstm_network->nodes_count_per_cell; node_index++) {
			lstm_network->full_mean_squared_error = lstm_network->full_mean_squared_error + pow(lstm_network->lstm_cells[cell_index].hidden_state[node_index] - lstm_network->lstm_cells[cell_index].expected_outputs[node_index], 2);
		}
	}
	lstm_network->full_mean_squared_error = lstm_network->full_mean_squared_error / (lstm_network->cells_count * lstm_network->nodes_count_per_cell);
}

void lstm_neural_network_mean_squared_error_calculation(t_lstm_neural_network *lstm_network, int from_cell_index) {
	lstm_network->mean_squared_error_from_index = 0;
	for (int cell_index = from_cell_index; cell_index < lstm_network->cells_count; cell_index++) {
		for (int node_index = 0; node_index < lstm_network->nodes_count_per_cell; node_index++) {
			lstm_network->mean_squared_error_from_index = lstm_network->mean_squared_error_from_index + pow(lstm_network->lstm_cells[cell_index].hidden_state[node_index] - lstm_network->lstm_cells[cell_index].expected_outputs[node_index], 2);
		}
	}
	lstm_network->mean_squared_error_from_index = lstm_network->mean_squared_error_from_index / ((lstm_network->cells_count - from_cell_index) * lstm_network->nodes_count_per_cell);
}

void lstm_neural_network_prepare_direction(t_lstm_neural_network *lstm_network) {
	for (int cell_index = 0; cell_index < lstm_network->cells_count; cell_index++) {
		printf("prepare cell %d\n", cell_index);
		for (int node_index = 0; node_index < lstm_network->lstm_cells[cell_index].node_count_per_single_gate; node_index++) {
			lstm_neural_network_mean_squared_error_calculation(lstm_network, cell_index);
			lstm_network->mean_squared_error_from_index_temp = lstm_network->mean_squared_error_from_index;
			for (int weight_index = 0; weight_index < lstm_network->lstm_cells[cell_index].forget_gate->nodes[node_index].inputCount; weight_index++) {
				node_set_direction(&lstm_network->lstm_cells[cell_index].forget_gate->nodes[node_index], weight_index, POSITIVE);
				node_action_with_saving_weight(&lstm_network->lstm_cells[cell_index].forget_gate->nodes[node_index], weight_index);
				lstm_neural_network_forward_propagation_from_cell_index(lstm_network, cell_index);
				lstm_neural_network_mean_squared_error_calculation(lstm_network, cell_index);
				if (lstm_network->mean_squared_error_from_index_temp < lstm_network->mean_squared_error_from_index) {
					node_set_direction(&lstm_network->lstm_cells[cell_index].forget_gate->nodes[node_index], weight_index, NEGATIVE);
				}
				node_weight_recover(&lstm_network->lstm_cells[cell_index].forget_gate->nodes[node_index], weight_index);
			}
		}
		for (int node_index = 0; node_index < lstm_network->lstm_cells[cell_index].input_gate->nodesCount; node_index++) {
			for (int weight_index = 0; weight_index < lstm_network->lstm_cells[cell_index].input_gate->nodes[node_index].inputCount; weight_index++) {
				node_set_direction(&lstm_network->lstm_cells[cell_index].input_gate->nodes[node_index], weight_index, POSITIVE);
				node_action_with_saving_weight(&lstm_network->lstm_cells[cell_index].input_gate->nodes[node_index], weight_index);
				lstm_neural_network_forward_propagation_from_cell_index(lstm_network, cell_index);
				lstm_neural_network_mean_squared_error_calculation(lstm_network, cell_index);
				if (lstm_network->mean_squared_error_from_index_temp < lstm_network->mean_squared_error_from_index) {
					node_set_direction(&lstm_network->lstm_cells[cell_index].input_gate->nodes[node_index], weight_index, NEGATIVE);
				}
				node_weight_recover(&lstm_network->lstm_cells[cell_index].input_gate->nodes[node_index], weight_index);
			}
		}
		for (int node_index = 0; node_index < lstm_network->lstm_cells[cell_index].candidate_cell_state_gate->nodesCount; node_index++) {
			for (int weight_index = 0; weight_index < lstm_network->lstm_cells[cell_index].candidate_cell_state_gate->nodes[node_index].inputCount; weight_index++) {
				node_set_direction(&lstm_network->lstm_cells[cell_index].candidate_cell_state_gate->nodes[node_index], weight_index, POSITIVE);
				node_action_with_saving_weight(&lstm_network->lstm_cells[cell_index].candidate_cell_state_gate->nodes[node_index], weight_index);
				lstm_neural_network_forward_propagation_from_cell_index(lstm_network, cell_index);
				lstm_neural_network_mean_squared_error_calculation(lstm_network, cell_index);
				if (lstm_network->mean_squared_error_from_index_temp < lstm_network->mean_squared_error_from_index) {
					node_set_direction(&lstm_network->lstm_cells[cell_index].candidate_cell_state_gate->nodes[node_index], weight_index, NEGATIVE);
				}
				node_weight_recover(&lstm_network->lstm_cells[cell_index].candidate_cell_state_gate->nodes[node_index], weight_index);
			}
		}
		for (int node_index = 0; node_index < lstm_network->lstm_cells[cell_index].output_gate->nodesCount; node_index++) {
			for (int weight_index = 0; weight_index < lstm_network->lstm_cells[cell_index].output_gate->nodes[node_index].inputCount; weight_index++) {
				node_set_direction(&lstm_network->lstm_cells[cell_index].output_gate->nodes[node_index], weight_index, POSITIVE);
				node_action_with_saving_weight(&lstm_network->lstm_cells[cell_index].output_gate->nodes[node_index], weight_index);
				lstm_neural_network_forward_propagation_from_cell_index(lstm_network, cell_index);
				lstm_neural_network_mean_squared_error_calculation(lstm_network, cell_index);
				if (lstm_network->mean_squared_error_from_index_temp < lstm_network->mean_squared_error_from_index) {
					node_set_direction(&lstm_network->lstm_cells[cell_index].output_gate->nodes[node_index], weight_index, NEGATIVE);
				}
				node_weight_recover(&lstm_network->lstm_cells[cell_index].output_gate->nodes[node_index], weight_index);
			}
		}
	}
}

void lstm_neural_network_learning_step(t_lstm_neural_network *lstm_network) {
	lstm_neural_network_forward_propagation(lstm_network);
	lstm_neural_network_prepare_direction(lstm_network);
	for (int cell_index = 0; cell_index < lstm_network->cells_count; cell_index++) {
		printf("learning cell %d\n", cell_index);
		for (int node_index = 0; node_index < lstm_network->lstm_cells[cell_index].node_count_per_single_gate; node_index++) {
			lstm_neural_network_mean_squared_error_calculation(lstm_network, cell_index);
			lstm_network->mean_squared_error_from_index_temp = lstm_network->mean_squared_error_from_index;
			for (int weight_index = 0; weight_index < lstm_network->lstm_cells[cell_index].forget_gate->nodes[node_index].inputCount; weight_index++) {
				node_action_with_saving_weight(&lstm_network->lstm_cells[cell_index].forget_gate->nodes[node_index], weight_index);
			}
		}
		for (int node_index = 0; node_index < lstm_network->lstm_cells[cell_index].input_gate->nodesCount; node_index++) {
			for (int weight_index = 0; weight_index < lstm_network->lstm_cells[cell_index].input_gate->nodes[node_index].inputCount; weight_index++) {
				node_action_with_saving_weight(&lstm_network->lstm_cells[cell_index].input_gate->nodes[node_index], weight_index);
			}
		}
		for (int node_index = 0; node_index < lstm_network->lstm_cells[cell_index].candidate_cell_state_gate->nodesCount; node_index++) {
			for (int weight_index = 0; weight_index < lstm_network->lstm_cells[cell_index].candidate_cell_state_gate->nodes[node_index].inputCount; weight_index++) {
				node_action_with_saving_weight(&lstm_network->lstm_cells[cell_index].candidate_cell_state_gate->nodes[node_index], weight_index);
			}
		}
		for (int node_index = 0; node_index < lstm_network->lstm_cells[cell_index].output_gate->nodesCount; node_index++) {
			for (int weight_index = 0; weight_index < lstm_network->lstm_cells[cell_index].output_gate->nodes[node_index].inputCount; weight_index++) {
				node_action_with_saving_weight(&lstm_network->lstm_cells[cell_index].output_gate->nodes[node_index], weight_index);
			}
		}
	}
}

void lstm_neural_network_destroy(t_lstm_neural_network *lstm_network) {
	free(lstm_network->lstm_cells);
}
