//
// Created by User on 16.06.2026.
//

#include "lstm_cell.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

void lstm_cell_init(t_lstm_cell *lstm_cell, int inputs_count, int node_count_per_single_gate){
	lstm_cell->inputs_count = inputs_count;
	lstm_cell->node_count_per_single_gate = node_count_per_single_gate;
	lstm_cell->state_vectors_size = node_count_per_single_gate;
	lstm_cell->inputs_to_layers_count_without_biases = inputs_count + node_count_per_single_gate;
	lstm_cell->inputs = malloc(sizeof(double) * inputs_count);
	lstm_cell->hidden_state_inputs = malloc(sizeof(double) * node_count_per_single_gate);
	lstm_cell->cell_state_inputs = malloc(sizeof(double) * node_count_per_single_gate);
	lstm_cell->temp_state_vector = malloc(sizeof(double) * node_count_per_single_gate);
	lstm_cell->temp_single_double_value = malloc(sizeof(double));
	for (int state_index = 0; state_index < node_count_per_single_gate; state_index++) {
		lstm_cell->hidden_state_inputs[state_index] = 0.0;
		lstm_cell->cell_state_inputs[state_index] = 0.0;
	}
	lstm_cell->hidden_state = malloc(sizeof(double) * node_count_per_single_gate);
	lstm_cell->cell_state = malloc(sizeof(double) * node_count_per_single_gate);
	lstm_cell->expected_vector = malloc(sizeof(double) * node_count_per_single_gate);
	lstm_cell->inputs_to_layers = malloc(sizeof(double) * lstm_cell->inputs_to_layers_count_without_biases);
	lstm_cell->forget_gate = malloc(sizeof(t_layer));
	lstm_cell->input_gate = malloc(sizeof(t_layer));
	lstm_cell->candidate_cell_state_gate = malloc(sizeof(t_layer));
	lstm_cell->output_gate = malloc(sizeof(t_layer));
	layer_init(lstm_cell->forget_gate, lstm_cell->inputs_count + lstm_cell->node_count_per_single_gate, lstm_cell->node_count_per_single_gate, BIASES_COUNT);
	layer_init(lstm_cell->input_gate, lstm_cell->inputs_count + lstm_cell->node_count_per_single_gate, lstm_cell->node_count_per_single_gate, BIASES_COUNT);
	layer_init(lstm_cell->candidate_cell_state_gate, lstm_cell->inputs_count + lstm_cell->node_count_per_single_gate, lstm_cell->node_count_per_single_gate, BIASES_COUNT);
	layer_init(lstm_cell->output_gate, lstm_cell->inputs_count + lstm_cell->node_count_per_single_gate, lstm_cell->node_count_per_single_gate, BIASES_COUNT);
}

void lstm_cell_set_cell_state_inputs(t_lstm_cell *lstm_cell, double *inputs) {
	memcpy(lstm_cell->cell_state_inputs, inputs, sizeof(double) * lstm_cell->state_vectors_size);
}

void lstm_cell_set_hidden_state_inputs(t_lstm_cell *lstm_cell, double *inputs) {
	memcpy(lstm_cell->hidden_state_inputs, inputs, sizeof(double) * lstm_cell->state_vectors_size);
}

void lstm_cell_set_inputs(t_lstm_cell *lstm_cell, double *inputs) {
	memcpy(lstm_cell->inputs, inputs, sizeof(double) * lstm_cell->inputs_count);
}

void lstm_cell_set_expected_vector(t_lstm_cell *lstm_cell, double *inputs) {
	memcpy(lstm_cell->expected_vector, inputs, sizeof(double) * lstm_cell->state_vectors_size);
}

void lstm_cell_calculate_all_gates(t_lstm_cell *lstm_cell) {
	memcpy(lstm_cell->inputs_to_layers, lstm_cell->hidden_state_inputs, sizeof(double) * lstm_cell->state_vectors_size);
	memcpy(&lstm_cell->inputs_to_layers[lstm_cell->state_vectors_size], lstm_cell->inputs, sizeof(double) * lstm_cell->inputs_count);
	layer_set_inputs(lstm_cell->forget_gate, lstm_cell->inputs_to_layers);
	layer_set_inputs(lstm_cell->input_gate, lstm_cell->inputs_to_layers);
	layer_set_inputs(lstm_cell->candidate_cell_state_gate, lstm_cell->inputs_to_layers);
	layer_set_inputs(lstm_cell->output_gate, lstm_cell->inputs_to_layers);
	layer_calculate_sigma_output(lstm_cell->forget_gate);
	layer_calculate_sigma_output(lstm_cell->input_gate);
	layer_calculate_tanh_output(lstm_cell->candidate_cell_state_gate);
	layer_calculate_sigma_output(lstm_cell->output_gate);
}

void lstm_cell_hadamard_product(double *a, double *b, double *dest_vector, int size) {
	for (int vector_index = 0; vector_index < size; ++vector_index) {
		dest_vector[vector_index] = a[vector_index] * b[vector_index];
	}
}

void lstm_cell_pointwise_addition(double *a, double *b, double *dest_vector, int size) {
	for (int vector_index = 0; vector_index < size; ++vector_index) {
		dest_vector[vector_index] = a[vector_index] + b[vector_index];
	}
}

void lstm_cell_tanh_cell_state_to_temp_vector(t_lstm_cell *lstm_cell) {
	for (int vector_index = 0; vector_index < lstm_cell->state_vectors_size; ++vector_index) {
		*lstm_cell->temp_single_double_value = exp(2.0 * lstm_cell->cell_state[vector_index]);
		lstm_cell->temp_state_vector[vector_index] = (*lstm_cell->temp_single_double_value - 1.0) / (*lstm_cell->temp_single_double_value + 1.0);
	}
}

void lstm_cell_forward_propagation(t_lstm_cell *lstm_cell) {
	lstm_cell_calculate_all_gates(lstm_cell);
	lstm_cell_hadamard_product(lstm_cell->forget_gate->output, lstm_cell->cell_state_inputs, lstm_cell->cell_state, lstm_cell->state_vectors_size);
	lstm_cell_hadamard_product(lstm_cell->input_gate->output, lstm_cell->candidate_cell_state_gate->output, lstm_cell->temp_state_vector, lstm_cell->state_vectors_size);
	lstm_cell_pointwise_addition(lstm_cell->cell_state, lstm_cell->temp_state_vector, lstm_cell->cell_state, lstm_cell->state_vectors_size);
	lstm_cell_tanh_cell_state_to_temp_vector(lstm_cell);
	lstm_cell_hadamard_product(lstm_cell->output_gate->output, lstm_cell->temp_state_vector, lstm_cell->hidden_state, lstm_cell->state_vectors_size);
}

void lstm_cell_destroy(t_lstm_cell *lstm_cell) {
	free(lstm_cell->inputs);
	free(lstm_cell->hidden_state_inputs);
	free(lstm_cell->cell_state_inputs);
	free(lstm_cell->hidden_state);
	free(lstm_cell->temp_state_vector);
	free(lstm_cell->temp_single_double_value);
	free(lstm_cell->cell_state);
	free(lstm_cell->expected_vector);
	free(lstm_cell->inputs_to_layers);
	layer_destroy(lstm_cell->forget_gate);
	layer_destroy(lstm_cell->input_gate);
	layer_destroy(lstm_cell->candidate_cell_state_gate);
	layer_destroy(lstm_cell->output_gate);
	free(lstm_cell->forget_gate);
	free(lstm_cell->input_gate);
	free(lstm_cell->candidate_cell_state_gate);
	free(lstm_cell->output_gate);
}