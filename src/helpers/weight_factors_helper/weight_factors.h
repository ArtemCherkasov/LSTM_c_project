//
// Created by User on 28.06.2026.
//
#include "../../nn/lstm/lstm_neural_network/lstm_neural_network.h"
#include "../../../src/helpers/main_struct/main_struct.h"

#ifndef LSTM_C_PROJECT_WEIGHT_FACTORS_H
#define LSTM_C_PROJECT_WEIGHT_FACTORS_H

typedef struct WeightFactorsHeader t_weight_factors_header;

struct WeightFactorsHeader {
	char price_symbol[10];
	char full_mean_squared_error[20];
	char cell_count[10];
	char node_count_per_cell[10];
};

void weight_factors_save_to_file(t_lstm_neural_network *lstm_network, t_main_struct *main_struct);
void weight_factors_load_from_file(t_lstm_neural_network *lstm_network, t_main_struct *main_struct);

#endif //LSTM_C_PROJECT_WEIGHT_FACTORS_H
