#include <math.h>
#include <stdio.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "helpers/mt5_file_read/mt5_file_read.h"
#include "helpers/main_struct/main_struct.h"
#include "helpers/weight_factors_helper/weight_factors.h"
#include "mode/forecast/forecast_mode.h"
#include "mode/pure_price/pure_price_mode.h"
#include "mode/test/test_mode.h"
#include "mode/training/training_mode.h"
#include "nn/lstm/lstm_neural_network/lstm_neural_network.h"

t_main_struct *main_struct;
t_lstm_neural_network *lstm_network;
t_lstm_neural_network *lstm_network_first_pointer;
t_lstm_neural_network *lstm_network_last_pointer;

void exit_handler(int n_signal) {
	printf("\nCODE %d\n", n_signal);
	if ((strcmp(main_struct->mode, "training") == 0) && main_struct->weight_factors_file_path != 0) {
		printf("\nSave weight factors to %s\n", main_struct->weight_factors_file_path);
		weight_factors_save_to_file(lstm_network, main_struct);
	}
	printf("\nExiting ...\n");
	raise(SIGTERM);
}

int main(int argc, char *argv[]) {
	main_struct = malloc(sizeof(t_main_struct));
	prepare_main_struct(main_struct, argc, argv);

	signal(SIGINT, &exit_handler);
	srand(time(NULL));
	lstm_network = malloc(sizeof(t_lstm_neural_network));
	if (main_struct->step_forecasts > 0) {
		lstm_neural_network_init_with_not_empty_tail(lstm_network, main_struct->cell_count, INPUT_COUNT_PER_CELL, PREDICT_VECTOR_SIZE, main_struct->step_forecasts, INPUT_COUNT_PER_TAIL_CELL);
	} else {
		lstm_neural_network_init(lstm_network, main_struct->cell_count, INPUT_COUNT_PER_CELL, PREDICT_VECTOR_SIZE);
	}

	lstm_network->index = 0;
	lstm_network_first_pointer = lstm_network;
	lstm_network_last_pointer = lstm_network;

	if (main_struct->layers_count > 1) {
		for (int layer_index = 1; layer_index < main_struct->layers_count; layer_index++) {
			lstm_network->next = malloc(sizeof(t_lstm_neural_network));
			lstm_neural_network_init(lstm_network->next, main_struct->cell_count + main_struct->step_forecasts, PREDICT_VECTOR_SIZE, PREDICT_VECTOR_SIZE);
			lstm_network->next->index = layer_index;
			lstm_network->next->prev = lstm_network;
			lstm_network = lstm_network->next;
			lstm_network_last_pointer = lstm_network;
		}
		lstm_network = lstm_network_first_pointer;
	}

	main_struct->file = malloc(sizeof(t_mt5file));

	if (strcmp(main_struct->mode, "test") == 0) {
		test_process(lstm_network, lstm_network_first_pointer, lstm_network_last_pointer, main_struct);
	} else if (strcmp(main_struct->mode, "training") == 0) {
		training_process(lstm_network, lstm_network_first_pointer, lstm_network_last_pointer, main_struct);
	} else if (strcmp(main_struct->mode, "forecast") == 0) {
		print_forecast_to_standard_output(lstm_network, lstm_network_first_pointer, lstm_network_last_pointer, main_struct);
	} else if (strcmp(main_struct->mode, "price") == 0) {
		print_pure_price_to_standard_output(main_struct);
	}

	lstm_neural_network_destroy(lstm_network_first_pointer);
	mt5_file_destroy(main_struct->file);
	free(main_struct->file);
	free(main_struct);
	return 0;
}
