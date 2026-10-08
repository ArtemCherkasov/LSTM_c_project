//
// Created by User on 28.06.2026.
//

#include "main_struct.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void prepare_main_struct(t_main_struct *main_struct, int argc, char *argv[]) {

	main_struct->source_file = 0;
	main_struct->weight_factors_file_path = 0;
	main_struct->price_symbol = 0;
	main_struct->learning_rate = 0;
	main_struct->layers_count = 1;
	main_struct->step_forecasts = STEP_FORECAST;
	main_struct->forecasts_gap = FORECAST_GAP;
	main_struct->cell_count = CELL_COUNT;
	main_struct->verbose = false;

	for (int arg_index = 0; arg_index < argc; arg_index++) {
		if (strcmp(argv[arg_index], "-sr") == 0 || strcmp(argv[arg_index], "--source") == 0) {
			main_struct->source_file = argv[arg_index + 1];
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
		if (strcmp(argv[arg_index], "--getfromline") == 0) {
			main_struct->get_from_line = atoi(argv[arg_index + 1]);
		}
		if (strcmp(argv[arg_index], "--getcount") == 0) {
			main_struct->get_count = atoi(argv[arg_index + 1]);
		}
		if (strcmp(argv[arg_index], "-v") == 0 || strcmp(argv[arg_index], "--verbose") == 0) {
			main_struct->verbose = true;
		}
	}
}

void print_main_struct_info(t_main_struct *main_struct) {
	if (main_struct->verbose) {
		printf("main struct info\n");
		printf("mode %s\n", main_struct->mode);
		printf("source file %s\n", main_struct->source_file);
		printf("forecast from line %d\n", main_struct->forecast_from_line);
		printf("weight factors file path %s\n", main_struct->weight_factors_file_path);
		printf("price symbol %s\n", main_struct->price_symbol);
		printf("learning rate %1.3f\n", main_struct->learning_rate);
		printf("layers count %d\n", main_struct->layers_count);
		printf("step forecast %d\n", main_struct->step_forecasts);
		printf("forecast gap %d\n", main_struct->forecasts_gap);
		printf("cells count %d\n", main_struct->cell_count);
	}
}