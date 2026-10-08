//
// Created by User on 07.10.2026.
//

#include "training_mode.h"
#include <stdio.h>

void training_process(t_lstm_neural_network *lstm_network, t_lstm_neural_network *lstm_network_first_pointer, t_lstm_neural_network *lstm_network_last_pointer, t_main_struct *main_struct) {
	/* Training mode
	 * acceptable mean square error MSE 0.0000000002
	 */
	printf("\nTraining mode:\n");
	mt5_file_init(main_struct->file, main_struct->source_file);
	if (main_struct->weight_factors_file_path != 0) {
		weight_factors_load_from_file(lstm_network, main_struct);
	}
	if (main_struct->learning_rate != 0) {
		printf("Set learning rate: %0.5f\n", main_struct->learning_rate);
		lstm_network->learning_rate = main_struct->learning_rate;
		if (main_struct->layers_count > 1) {
			for (int layer_index = 1; layer_index < main_struct->layers_count; layer_index++) {
				lstm_network->next->learning_rate = main_struct->learning_rate;
				lstm_network = lstm_network->next;
			}
			lstm_network = lstm_network_first_pointer;
		}
	}

	int file_row_pointer = 1000;
	int file_finish_row_pointer = 90000;

	for (int row_index = file_row_pointer; row_index < file_finish_row_pointer; row_index++) {
		for (int cell_index = 0; cell_index < main_struct->cell_count; cell_index++) {
			lstm_cell_set_inputs(&lstm_network->lstm_cells[cell_index], main_struct->file->lines[row_index + cell_index].primary_cell_buffer);
		}
		for (int cell_index = main_struct->cell_count; cell_index < (main_struct->cell_count + main_struct->step_forecasts); cell_index++) {
			lstm_cell_set_inputs(&lstm_network->lstm_cells[cell_index], main_struct->file->lines[row_index + cell_index].tail_cell_buffer);
		}
		for (int cell_index = 0; cell_index < (main_struct->cell_count + main_struct->step_forecasts); cell_index++) {
			lstm_cell_set_expected_vector(&lstm_network_last_pointer->lstm_cells[cell_index], main_struct->file->lines[row_index + cell_index + main_struct->forecasts_gap].short_buffer_diff);
		}
		for (int batch_index = 0; batch_index < 50; batch_index++) {
			lstm_neural_network_learning_step_bptt(lstm_network);
			lstm_neural_network_forward_propagation(lstm_network);
		}
		lstm_neural_network_full_mean_squared_error_calculation(lstm_network);
		printf("file pointer %d\n", row_index);
		printf("MSE %3.15f\n", lstm_network_last_pointer->full_mean_squared_error);
		if (main_struct->source_file != 0 && main_struct->weight_factors_file_path != 0) {
			printf("\nSave weight factors to %s\n", main_struct->weight_factors_file_path);
			weight_factors_save_to_file(lstm_network, main_struct);
		}
	}
}
