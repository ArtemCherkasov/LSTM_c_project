//
// Created by User on 28.06.2026.
//

#include "weight_factors.h"

#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "../mt5_file_read/mt5_file_read.h"

#define WEIGHT_DELIMITER " "

void weight_factors_save_to_file(t_lstm_neural_network *lstm_network, t_main_struct *main_struct) {
	FILE *file;
	char path[PATH_MAX];
	char *file_separator = malloc(sizeof(char));
	t_lstm_neural_network *lstm_network_first_pointer;
	t_lstm_neural_network *lstm_network_last_pointer;

	strcpy(file_separator, FILE_SEPARATOR);
	if (strcmp(&main_struct->weight_factors_file_path[0], file_separator) == 0) {
		strcat(path, main_struct->weight_factors_file_path);
	} else {
		getcwd(path, sizeof(path));
		strcat(path, file_separator);
		strcat(path, main_struct->weight_factors_file_path);
	}
	file = fopen(path, "w");
	if (file == NULL) {
		printf("\nCould not save file %s\n", path);
		return;
	}
	lstm_network_first_pointer = lstm_network;
	while (lstm_network->next != NULL) {
		lstm_network = lstm_network->next;
	}
	lstm_network_last_pointer = lstm_network;
	lstm_network = lstm_network_first_pointer;
	printf("\nSave to %s\n", path);
	if (main_struct->price_symbol != 0) {
		fprintf(file, "%s\n", main_struct->price_symbol);
	} else {
		fprintf(file, "NOT SYMBOL\n");
	}
	fprintf(file, "ITERATION %d\n", lstm_network->learning_step_index);
	fprintf(file, "MSE %3.20f\n", lstm_network_last_pointer->full_mean_squared_error);
	fprintf(file, "CELLS %d\n", lstm_network->cells_count);
	fprintf(file, "NODES PER CELL %d\n", lstm_network->nodes_count_per_cell);
	/*
	 *save weight
	 */
	while (lstm_network != NULL) {
		for (int cell_index = 0; cell_index < lstm_network->cells_count; cell_index++) {
			for (int node_index = 0; node_index < lstm_network->lstm_cells[cell_index].node_count_per_single_gate; node_index++) {
				lstm_neural_network_mean_squared_error_calculation(lstm_network, cell_index);
				lstm_network->mean_squared_error_from_index_temp_for_negative_direction = lstm_network->mean_squared_error_from_index;
				for (int weight_index = 0; weight_index < lstm_network->lstm_cells[cell_index].forget_gate->nodes[node_index].inputCount; weight_index++) {
					fprintf(file, "%3.20f ", lstm_network->lstm_cells[cell_index].forget_gate->nodes[node_index].weights[weight_index]);
				}
				fprintf(file, "\n");
			}
			for (int node_index = 0; node_index < lstm_network->lstm_cells[cell_index].input_gate->nodesCount; node_index++) {
				for (int weight_index = 0; weight_index < lstm_network->lstm_cells[cell_index].input_gate->nodes[node_index].inputCount; weight_index++) {
					fprintf(file, "%3.20f ", lstm_network->lstm_cells[cell_index].input_gate->nodes[node_index].weights[weight_index]);
				}
				fprintf(file, "\n");
			}
			for (int node_index = 0; node_index < lstm_network->lstm_cells[cell_index].candidate_cell_state_gate->nodesCount; node_index++) {
				for (int weight_index = 0; weight_index < lstm_network->lstm_cells[cell_index].candidate_cell_state_gate->nodes[node_index].inputCount; weight_index++) {
					fprintf(file, "%3.20f ", lstm_network->lstm_cells[cell_index].candidate_cell_state_gate->nodes[node_index].weights[weight_index]);
				}
				fprintf(file, "\n");
			}
			for (int node_index = 0; node_index < lstm_network->lstm_cells[cell_index].output_gate->nodesCount; node_index++) {
				for (int weight_index = 0; weight_index < lstm_network->lstm_cells[cell_index].output_gate->nodes[node_index].inputCount; weight_index++) {
					fprintf(file, "%3.20f ", lstm_network->lstm_cells[cell_index].output_gate->nodes[node_index].weights[weight_index]);
				}
				fprintf(file, "\n");
			}
		}
		lstm_network = lstm_network->next;
	}

	fclose(file);
	free(file_separator);
}

void weight_factors_load_from_file(t_lstm_neural_network *lstm_network, t_main_struct *main_struct) {
	FILE *file;
	char buffer[1024];
	char *token_weight;
	char weight_delimiter[] = {WEIGHT_DELIMITER};
	char path[PATH_MAX];
	char *file_separator = malloc(sizeof(char));

	strcpy(file_separator, FILE_SEPARATOR);
	if (strcmp(&main_struct->weight_factors_file_path[0], file_separator) == 0) {
		strcat(path, main_struct->weight_factors_file_path);
	} else {
		getcwd(path, sizeof(path));
		strcat(path, file_separator);
		strcat(path, main_struct->weight_factors_file_path);
	}
	file = fopen(path, "r");
	if (file == NULL) {
		printf("\nCould not open file %s\n", path);
		free(file_separator);
		return;
	}
	printf("\nOpen file %s\n", path);

	/*
	 *load weight
	 */
	fgets(buffer, sizeof(char) * 1024, file);
	fgets(buffer, sizeof(char) * 1024, file);
	token_weight = strtok(buffer, weight_delimiter);
	token_weight = strtok(NULL, weight_delimiter);
	lstm_network->learning_step_index = atoi(token_weight);
	fgets(buffer, sizeof(char) * 1024, file);
	fgets(buffer, sizeof(char) * 1024, file);
	fgets(buffer, sizeof(char) * 1024, file);
	while (lstm_network != NULL) {
		for (int cell_index = 0; cell_index < lstm_network->cells_count; cell_index++) {
			for (int node_index = 0; node_index < lstm_network->lstm_cells[cell_index].node_count_per_single_gate; node_index++) {
				fgets(buffer, sizeof(char) * 1024, file);
				token_weight = strtok(buffer, weight_delimiter);
				for (int weight_index = 0; weight_index < lstm_network->lstm_cells[cell_index].forget_gate->nodes[node_index].inputCount; weight_index++) {
					if (token_weight != 0) {
						lstm_network->lstm_cells[cell_index].forget_gate->nodes[node_index].weights[weight_index] = atof(token_weight);
					} else {
						printf("Load error: cell %d, forget gate, node %d, weight %d\n", cell_index, node_index, weight_index);
					}
					token_weight = strtok(NULL, weight_delimiter);
				}
			}
			for (int node_index = 0; node_index < lstm_network->lstm_cells[cell_index].input_gate->nodesCount; node_index++) {
				fgets(buffer, sizeof(char) * 1024, file);
				token_weight = strtok(buffer, weight_delimiter);
				for (int weight_index = 0; weight_index < lstm_network->lstm_cells[cell_index].input_gate->nodes[node_index].inputCount; weight_index++) {
					if (token_weight != 0) {
						lstm_network->lstm_cells[cell_index].input_gate->nodes[node_index].weights[weight_index] = atof(token_weight);
					} else {
						printf("Load error: cell %d, input gate, node %d, weight %d\n", cell_index, node_index, weight_index);
					}
					token_weight = strtok(NULL, weight_delimiter);
				}
			}
			for (int node_index = 0; node_index < lstm_network->lstm_cells[cell_index].candidate_cell_state_gate->nodesCount; node_index++) {
				fgets(buffer, sizeof(char) * 1024, file);
				token_weight = strtok(buffer, weight_delimiter);
				for (int weight_index = 0; weight_index < lstm_network->lstm_cells[cell_index].candidate_cell_state_gate->nodes[node_index].inputCount; weight_index++) {
					if (token_weight != 0) {
						lstm_network->lstm_cells[cell_index].candidate_cell_state_gate->nodes[node_index].weights[weight_index] = atof(token_weight);
					} else {
						printf("Load error: cell %d, candidate cell gate, node %d, weight %d\n", cell_index, node_index, weight_index);
					}
					token_weight = strtok(NULL, weight_delimiter);
				}
			}
			for (int node_index = 0; node_index < lstm_network->lstm_cells[cell_index].output_gate->nodesCount; node_index++) {
				fgets(buffer, sizeof(char) * 1024, file);
				token_weight = strtok(buffer, weight_delimiter);
				for (int weight_index = 0; weight_index < lstm_network->lstm_cells[cell_index].output_gate->nodes[node_index].inputCount; weight_index++) {
					if (token_weight != 0) {
						lstm_network->lstm_cells[cell_index].output_gate->nodes[node_index].weights[weight_index] = atof(token_weight);
					} else {
						printf("Load error: cell %d, output gate, node %d, weight %d\n", cell_index, node_index, weight_index);
					}
					token_weight = strtok(NULL, weight_delimiter);
				}
			}
		}
		lstm_network = lstm_network->next;
	}
	fclose(file);
	free(file_separator);
}
