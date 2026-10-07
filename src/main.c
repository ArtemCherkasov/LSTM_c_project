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
#include "mode/test/test_mode.h"
#include "mode/training/training_mode.h"
#include "nn/lstm/lstm_neural_network/lstm_neural_network.h"

#define DAYS 30
#define HOURS 24
#define CELL_COUNT (24*DAYS)
#define CELL_COUNT_TEST (24*DAYS)
#define STEP_FORECAST 0
#define STEP_FORECAST_TEST 0
#define FORECAST_GAP 1
#define FORECATS_GAP_TEST_MODE 1

t_main_struct *main_struct;
t_lstm_neural_network *lstm_network;
t_lstm_neural_network *lstm_network_first_pointer;
t_lstm_neural_network *lstm_network_last_pointer;

void exit_handler(int n_signal) {
	printf("\nCODE %d\n", n_signal);
	if (main_struct->training_source_file_path != 0 && main_struct->weight_factors_file_path != 0) {
		printf("\nSave weight factors to %s\n", main_struct->weight_factors_file_path);
		weight_factors_save_to_file(lstm_network, main_struct);
	}
	printf("\nExiting ...\n");
	raise(SIGTERM);
}

int main(int argc, char *argv[]) {
	main_struct = malloc(sizeof(t_main_struct));
	main_struct->training_source_file_path = 0;
	main_struct->source_to_forecast_file_path = 0;
	main_struct->weight_factors_file_path = 0;
	main_struct->price_symbol = 0;
	main_struct->learning_rate = 0;
	main_struct->test_mode = 0;
	main_struct->layers_count = 1;
	main_struct->step_forecasts = STEP_FORECAST;
	main_struct->forecasts_gap = FORECAST_GAP;
	main_struct->cell_count = CELL_COUNT;

	for (int arg_index = 0; arg_index < argc; arg_index++) {
		if (strcmp(argv[arg_index], "-ts") == 0 || strcmp(argv[arg_index], "--trainingsource") == 0) {
			main_struct->training_source_file_path = argv[arg_index + 1];
		}
		if (strcmp(argv[arg_index], "--test") == 0) {
			main_struct->cell_count = CELL_COUNT_TEST;
			main_struct->step_forecasts = STEP_FORECAST_TEST;
			main_struct->forecasts_gap = FORECATS_GAP_TEST_MODE;
			main_struct->test_mode = 1;
			main_struct->training_source_file_path = argv[arg_index + 1];
		}
		if (strcmp(argv[arg_index], "-fs") == 0 || strcmp(argv[arg_index], "--forecastsource") == 0) {
			main_struct->source_to_forecast_file_path = argv[arg_index + 1];
		}
		if (strcmp(argv[arg_index], "-ffl") == 0 || strcmp(argv[arg_index], "--forecastfromline") == 0) {
			main_struct->forecast_from_line = atoi(argv[arg_index + 1]);
		}
		if (strcmp(argv[arg_index], "-s") == 0 || strcmp(argv[arg_index], "--symbol") == 0) {
			main_struct->price_symbol = argv[arg_index + 1];
		}
		if (strcmp(argv[arg_index], "-w") == 0 || strcmp(argv[arg_index], "--weight") == 0) {
			main_struct->weight_factors_file_path = argv[arg_index + 1];
		}
		if (strcmp(argv[arg_index], "-lr") == 0 || strcmp(argv[arg_index], "--learningrate") == 0) {
			main_struct->learning_rate = atof(argv[arg_index + 1]);
		}
		if (strcmp(argv[arg_index], "-lc") == 0 || strcmp(argv[arg_index], "--layerscount") == 0) {
			main_struct->layers_count = atoi(argv[arg_index + 1]);
		}
		if (strcmp(argv[arg_index], "-cc") == 0 || strcmp(argv[arg_index], "--cellcount") == 0) {
			main_struct->cell_count = atoi(argv[arg_index + 1]);
		}
		if (strcmp(argv[arg_index], "-sf") == 0 || strcmp(argv[arg_index], "--stepforecast") == 0) {
			main_struct->step_forecasts = atoi(argv[arg_index + 1]);
		}
		if (strcmp(argv[arg_index], "-fg") == 0 || strcmp(argv[arg_index], "--forecastgap") == 0) {
			main_struct->forecasts_gap = atoi(argv[arg_index + 1]);
		}
		if (strcmp(argv[arg_index], "-m") == 0 || strcmp(argv[arg_index], "--mode") == 0) {
			main_struct->mode = argv[arg_index + 1];
		}
	}

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
	}

	lstm_neural_network_destroy(lstm_network_first_pointer);
	mt5_file_destroy(main_struct->file);
	free(main_struct->file);
	free(main_struct);
	return 0;
}
