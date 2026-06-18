//
// Created by User on 18.06.2026.
//

#include "lstm_neural_network.h"

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
		if (cell_index < lstm_network->cells_count) {
			lstm_cell_set_cell_state_inputs(&lstm_network->lstm_cells[cell_index + 1], lstm_network->lstm_cells[cell_index].cell_state);
			lstm_cell_set_hidden_state_inputs(&lstm_network->lstm_cells[cell_index + 1], lstm_network->lstm_cells[cell_index].hidden_state);
		}
	}
}

void lstm_neural_network_destroy(t_lstm_neural_network *lstm_network) {
	free(lstm_network->lstm_cells);
}
