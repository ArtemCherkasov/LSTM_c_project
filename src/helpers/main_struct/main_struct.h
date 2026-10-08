//
// Created by User on 28.06.2026.
//

#ifndef LSTM_C_PROJECT_MAIN_STRUCT_H
#define LSTM_C_PROJECT_MAIN_STRUCT_H
#include "../mt5_file_read/mt5_file_read.h"
typedef struct MainStruct t_main_struct;

struct MainStruct {
	int test_mode;
	char *mode;
	char *source_file;
	int forecast_from_line;
	char *weight_factors_file_path;
	char *price_symbol;
	double learning_rate;
	int layers_count;
	int step_forecasts;
	int forecasts_gap;
	int cell_count;
	int get_from_line;
	int get_count;
	bool verbose;
	t_mt5file *file;
};

void prepare_main_struct(t_main_struct *main_struct, int argc, char *argv[]);
void print_main_struct_info(t_main_struct *main_struct);

#endif //LSTM_C_PROJECT_MAIN_STRUCT_H
