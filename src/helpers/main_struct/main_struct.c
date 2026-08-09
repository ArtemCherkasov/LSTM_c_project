//
// Created by User on 28.06.2026.
//

#include "main_struct.h"

#include <stdio.h>

void print_main_struct_info(t_main_struct *main_struct) {
	printf("main struct info\n");
	printf("test mode %d\n", main_struct->test_mode);
	printf("training source file path %s\n", main_struct->training_source_file_path);
	printf("source to forecast file path %s\n", main_struct->source_to_forecast_file_path);
	printf("forecast from line %d\n", main_struct->forecast_from_line);
	printf("weight factors file path %s\n", main_struct->weight_factors_file_path);
	printf("price symbol %s\n", main_struct->price_symbol);
	printf("learning rate %1.3f\n", main_struct->learning_rate);
	printf("layers count %d\n", main_struct->layers_count);
	printf("step forecast %d\n", main_struct->step_forecasts);
	printf("forecast gap %d\n", main_struct->forecasts_gap);
	printf("cells count %d\n", main_struct->cell_count);
}
