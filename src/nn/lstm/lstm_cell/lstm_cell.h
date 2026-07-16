//
// Created by User on 16.06.2026.
//

#ifndef LSTM_C_PROJECT_LSTM_CELL_H
#define LSTM_C_PROJECT_LSTM_CELL_H
#include "../../common/layer/layer.h"

typedef struct LSTMCell t_lstm_cell;

struct LSTMCell {
	int cell_index;
	int node_count_per_single_gate;
	int state_vectors_size;
	int inputs_count;
	int inputs_to_layers_count_without_biases;
	t_layer *forget_gate;
	t_layer *input_gate;
	t_layer *candidate_cell_state_gate;
	t_layer *output_gate;
	double *cell_state_inputs;
	double *hidden_state_inputs;
	double *cell_state;
	double *hidden_state;
	double *temp_state_vector;
	double *temp_single_double_value;
	double *inputs;
	double *inputs_to_layers;
	double *expected_outputs;
	double *d_e_d_h;
	double *prev_d_e_d_hidden;
	double *prev_d_e_d_input;
	double *d_e_d_output_gate_vector;
	double *d_e_d_cell_state_vector;
	double *prev_d_e_d_cell_state_vector;
	double *d_e_d_input_gate_vector;
	double *d_e_d_candidate_cell_state_gate_vector;
	double *d_e_d_forget_gate_vector;
	double *cell_state_tanh;
	double *cell_state_tanh_derivative;
};

void lstm_cell_init(t_lstm_cell *lstm_cell, int inputs_count, int node_count_per_single_gate);
void lstm_cell_set_cell_state_inputs(t_lstm_cell *lstm_cell, double *inputs);
void lstm_cell_set_hidden_state_inputs(t_lstm_cell *lstm_cell, double *inputs);
void lstm_cell_set_inputs(t_lstm_cell *lstm_cell, double *inputs);
void lstm_cell_set_expected_vector(t_lstm_cell *lstm_cell, double *inputs);
void lstm_cell_calculate_all_gates(t_lstm_cell *lstm_cell);
void lstm_cell_hadamard_product(double *a, double *b, double *dest_vector, int size);
void lstm_cell_pointwise_addition(double *a, double *b, double *dest_vector, int size);
void lstm_cell_tanh_cell_state_to_temp_vector(t_lstm_cell *lstm_cell);
void lstm_cell_tanh_vector(double *src_vector, double *dest_vector, int size);
void lstm_cell_tanh_derivative_vector(double *src_vector, double *dest_vector, int size);
void lstm_cell_forward_propagation(t_lstm_cell *lstm_cell);
void lstm_cell_print_inputs(t_lstm_cell *lstm_cell);
void lstm_cell_print_any_vector(double *vector, int size);
void lstm_cell_destroy(t_lstm_cell *lstm_cell);
#endif //LSTM_C_PROJECT_LSTM_CELL_H
