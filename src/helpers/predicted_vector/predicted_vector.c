//
// Created by User on 11.07.2026.
//

#include "predicted_vector.h"

#include <math.h>
#include <stdlib.h>

void predicted_vector_init(t_predicted_vector *predicted_vector, int forecast_gap) {
	printf("predicted_vector_init\n");
	predicted_vector->predicted_vector_size = forecast_gap;
	predicted_vector->predicted_price = malloc(sizeof(t_predicted_price) * forecast_gap);
}

void predicted_vector_get_data_from_lstm_net(t_predicted_vector *predicted_vector, t_lstm_neural_network *lstm_network) {
	t_lstm_neural_network *lstm_network_first_pointer;
	t_lstm_neural_network *lstm_network_last_pointer;
	int predicted_price_index = 0;
	lstm_network_first_pointer = lstm_network;
	while (lstm_network->next != NULL) {
		lstm_network = lstm_network->next;
	}
	lstm_network_last_pointer = lstm_network;
	lstm_network = lstm_network_first_pointer;
	double open = lstm_network->lstm_cells[lstm_network->cells_count_without_forecast_cells - 1].inputs[OPEN_INDEX] * NORMALIZE_FACTOR_PRICE;
	double high = lstm_network->lstm_cells[lstm_network->cells_count_without_forecast_cells - 1].inputs[HIGH_INDEX] * NORMALIZE_FACTOR_PRICE;
	double low = lstm_network->lstm_cells[lstm_network->cells_count_without_forecast_cells - 1].inputs[LOW_INDEX] * NORMALIZE_FACTOR_PRICE;
	double close = lstm_network->lstm_cells[lstm_network->cells_count_without_forecast_cells - 1].inputs[CLOSE_INDEX] * NORMALIZE_FACTOR_PRICE;
	double open_next = 0.0;
	double high_next = 0.0;
	double low_next = 0.0;
	double close_next = 0.0;
	printf("last vector before predict [%2.10f %2.10f %2.10f %2.10f]\n", open, high, low, close);
	if (lstm_network_last_pointer->cells_count_full - predicted_vector->predicted_vector_size > 0) {
		for (int cell_index = (lstm_network_last_pointer->cells_count_full - predicted_vector->predicted_vector_size); cell_index < lstm_network_last_pointer->cells_count_full; cell_index++) {
			open_next = lstm_network_last_pointer->lstm_cells[cell_index - 1].hidden_state[SHORT_OPEN_INDEX];
			high_next = lstm_network_last_pointer->lstm_cells[cell_index - 1].hidden_state[SHORT_HIGH_INDEX];
			low_next = lstm_network_last_pointer->lstm_cells[cell_index - 1].hidden_state[SHORT_LOW_INDEX];
			close_next = lstm_network_last_pointer->lstm_cells[cell_index - 1].hidden_state[SHORT_CLOSE_INDEX];
			open = open - log((1 - open_next)/ open_next)/ AMOUNT_OF_EXPANSION_SIGMA;
			high = high - log((1 - high_next)/ high_next)/ AMOUNT_OF_EXPANSION_SIGMA;
			low = low - log((1 - low_next)/ low_next)/ AMOUNT_OF_EXPANSION_SIGMA;
			close = close - log((1 - close_next)/ close_next)/ AMOUNT_OF_EXPANSION_SIGMA;
			predicted_vector->predicted_price[predicted_price_index].open_output = open;
			predicted_vector->predicted_price[predicted_price_index].high_output = high;
			predicted_vector->predicted_price[predicted_price_index].low_output = low;
			predicted_vector->predicted_price[predicted_price_index].close_output = close;
			predicted_price_index++;
		}
	}
}

void predicted_vector_print(t_predicted_vector *predicted_vector) {
	for (int price_index = 0; price_index < predicted_vector->predicted_vector_size; price_index++) {
		printf("%d) [%2.10f %2.10f %2.10f %2.10f]\n", price_index, predicted_vector->predicted_price[price_index].open_output, predicted_vector->predicted_price[price_index].high_output, predicted_vector->predicted_price[price_index].low_output, predicted_vector->predicted_price[price_index].close_output);
	}
}

void predicted_vector_destroy(t_predicted_vector *predicted_vector) {
	free(predicted_vector->predicted_price);
}