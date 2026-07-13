//
// Created by User on 18.06.2026.
//

#ifndef LSTM_C_PROJECT_LSTM_NEURAL_NETWORK_H
#define LSTM_C_PROJECT_LSTM_NEURAL_NETWORK_H
#include "../lstm_cell/lstm_cell.h"

typedef struct LSTMNeuralNetwork t_lstm_neural_network;

struct LSTMNeuralNetwork {
	int index;
	int learning_step_index;
	int cells_count_without_forecast_cells;
	int cells_count_forecast;
	int cells_count_full;
	int inputs_count_per_cell;
	int nodes_count_per_cell;
	int inputs_count_per_network;
	int outputs_count_per_network;
	t_lstm_cell *lstm_cells;
	double *predicted_outputs;
	double *expected_outputs;
	double full_mean_squared_error;
	double mean_squared_error_mul_factor;
	double mean_squared_error_from_index;
	double mean_squared_error_from_index_temp_for_positive_direction;
	double mean_squared_error_from_index_temp_for_negative_direction;
	double learning_rate;
	t_lstm_neural_network *next;
};

void lstm_neural_network_init(t_lstm_neural_network *lstm_network, int cells_count, int inputs_count_per_cell, int nodes_count_per_cell);
void lstm_neural_network_init_with_empty_input_vector(t_lstm_neural_network *lstm_network, int cells_count, int inputs_count_per_cell, int nodes_count_per_cell, int cells_count_forecast);
void lstm_neural_network_set_input(t_lstm_neural_network *lstm_network, double* inputs);
void lstm_neural_network_forward_propagation(t_lstm_neural_network *lstm_network);
void lstm_neural_network_forward_propagation_from_cell_index(t_lstm_neural_network *lstm_network, int from_cell_index);
void lstm_neural_network_full_mean_squared_error_calculation(t_lstm_neural_network *lstm_network);
void lstm_neural_network_mean_squared_error_calculation(t_lstm_neural_network *lstm_network, int from_cell_index);
void lstm_neural_network_prepare_direction(t_lstm_neural_network *lstm_network);
void lstm_neural_network_learning_step(t_lstm_neural_network *lstm_network);
void lstm_neural_network_destroy(t_lstm_neural_network *lstm_network);
#endif //LSTM_C_PROJECT_LSTM_NEURAL_NETWORK_H
