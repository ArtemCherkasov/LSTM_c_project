//
// Created by User on 28.06.2026.
//

#ifndef LSTM_C_PROJECT_MAIN_STRUCT_H
#define LSTM_C_PROJECT_MAIN_STRUCT_H
typedef struct MainStruct t_main_struct;

struct MainStruct {
	int test_mode;
	char *training_source_file_path;
	char *source_to_forecast_file_path;
	int forecast_from_line;
	char *weight_factors_file_path;
	char *price_symbol;
	double learning_rate;
	int layers_count;
	int step_forecasts;
	int cell_count;
};

#endif //LSTM_C_PROJECT_MAIN_STRUCT_H
