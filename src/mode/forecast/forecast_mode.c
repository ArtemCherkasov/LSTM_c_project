//
// Created by User on 07.10.2026.
//

#include "forecast_mode.h"
#include <stdlib.h>
#include "../../helpers/main_struct/main_struct.h"
#include "../../helpers/format_output/format_output.h"

void print_forecast_to_standard_output(t_lstm_neural_network *lstm_network, t_lstm_neural_network *lstm_network_first_pointer, t_lstm_neural_network *lstm_network_last_pointer, t_main_struct *main_struct) {
	/*
	 * Forecast mode
	 */
	std_output_line(main_struct->verbose, "\nForecast mode:\n");
	print_main_struct_info(main_struct);
	t_predicted_vector *predicted_vector;
	mt5_file_init(main_struct->file, main_struct->source_file, main_struct->verbose);
	if (main_struct->weight_factors_file_path != 0) {
		std_output_line(main_struct->verbose, "Weight factors file: %s\n", main_struct->weight_factors_file_path);
		weight_factors_load_from_file(lstm_network, main_struct);
	} else {
		std_output_line(main_struct->verbose,"Error: file with weights must be defined! (use -w <FILE> or --weight <FILE>)\n");
	}

	predicted_vector = malloc(sizeof(t_predicted_vector));
	predicted_vector_init(predicted_vector, main_struct);
	predicted_price_init(&predicted_vector->predicted_price[0], main_struct);

	std_output_line(main_struct->verbose,"\n");
	int start_row = main_struct->forecast_from_line - main_struct->cell_count;
	int final_cell_index_before_predict = main_struct->cell_count - 1;
	std_output_line(main_struct->verbose,"start line from file %d\n", start_row);
	for (int cell_index = 0; cell_index < main_struct->cell_count; cell_index++) {
		lstm_cell_set_inputs(&lstm_network->lstm_cells[cell_index], main_struct->file->lines[start_row + cell_index].primary_cell_buffer);
	}
	for (int cell_index = main_struct->cell_count; cell_index < (main_struct->cell_count + main_struct->step_forecasts); cell_index++) {
		lstm_cell_set_inputs(&lstm_network->lstm_cells[cell_index], main_struct->file->lines[start_row + cell_index].tail_cell_buffer);
	}
	lstm_neural_network_forward_propagation(lstm_network);
	std_output_line(main_struct->verbose,"cell index %d, line in file %d\n[%3.15f %3.15f %3.15f %3.15f]\n", final_cell_index_before_predict, start_row + final_cell_index_before_predict, main_struct->file->lines[start_row + final_cell_index_before_predict].open, main_struct->file->lines[start_row + final_cell_index_before_predict].high, main_struct->file->lines[start_row + final_cell_index_before_predict].low, main_struct->file->lines[start_row + final_cell_index_before_predict].close);
	predicted_vector_get_data_from_lstm_net(predicted_vector, lstm_network_first_pointer);
	predicted_vector_print(predicted_vector);
	predicted_vector_destroy(predicted_vector);
}
