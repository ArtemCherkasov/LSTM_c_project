//
// Created by User on 11.07.2026.
//

#include "predicted_vector.h"

#include <stdlib.h>

void predicted_vector_init(t_predicted_vector *predicted_vector, int step_forecasts) {
	printf("predicted_vector_init\n");
	predicted_vector->predicted_vector_size = step_forecasts;
	predicted_vector->predicted_price = malloc(sizeof(t_predicted_price) * step_forecasts);
}

void predicted_vector_get_data_from_lstm_net(t_predicted_vector *predicted_vector, t_lstm_neural_network *lstm_network) {
	int predicted_price_index = 0;
	for (int cell_index = (lstm_network->cells_count_full - predicted_vector->predicted_vector_size); cell_index < lstm_network->cells_count_full; cell_index++) {
		if (predicted_price_index > 0) {
			predicted_vector->predicted_price[predicted_price_index].open = predicted_vector->predicted_price[predicted_price_index - 1].open_output;
			predicted_vector->predicted_price[predicted_price_index].high = predicted_vector->predicted_price[predicted_price_index - 1].high_output;
			predicted_vector->predicted_price[predicted_price_index].low = predicted_vector->predicted_price[predicted_price_index - 1].low_output;
			predicted_vector->predicted_price[predicted_price_index].close = predicted_vector->predicted_price[predicted_price_index - 1].close_output;
		}
		predicted_vector->predicted_price[predicted_price_index].open_output = predicted_vector->predicted_price[predicted_price_index].open + lstm_network->lstm_cells[cell_index].hidden_state[0] / NORMALIZE_FACTOR_DIFF;
		predicted_vector->predicted_price[predicted_price_index].high_output = predicted_vector->predicted_price[predicted_price_index].high + lstm_network->lstm_cells[cell_index].hidden_state[1] / NORMALIZE_FACTOR_DIFF;
		predicted_vector->predicted_price[predicted_price_index].low_output = predicted_vector->predicted_price[predicted_price_index].low + lstm_network->lstm_cells[cell_index].hidden_state[2] / NORMALIZE_FACTOR_DIFF;
		predicted_vector->predicted_price[predicted_price_index].close_output = predicted_vector->predicted_price[predicted_price_index].close + lstm_network->lstm_cells[cell_index].hidden_state[3] / NORMALIZE_FACTOR_DIFF;
		predicted_price_index++;
	}
}

void predicted_vector_print(t_predicted_vector *predicted_vector) {
	for (int price_index = 0; price_index < predicted_vector->predicted_vector_size; price_index++) {
		printf("%d) %2.10f %2.10f %2.10f %2.10f\n", price_index, predicted_vector->predicted_price[price_index].open_output, predicted_vector->predicted_price[price_index].high_output, predicted_vector->predicted_price[price_index].low_output, predicted_vector->predicted_price[price_index].close_output);
	}
}

void predicted_vector_destroy(t_predicted_vector *predicted_vector) {
	free(predicted_vector->predicted_price);
}