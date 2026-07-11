//
// Created by User on 11.07.2026.
//

#include "predicted_vector.h"

#include <stdlib.h>

void predicted_vector_init(t_predicted_vector *predicted_vector, int step_forecasts) {
	predicted_vector->predicted_price = malloc(sizeof(t_predicted_price) * step_forecasts);
}

void predicted_vector_set_date_time_to_price(t_predicted_vector *predicted_vector, double *date_time_vector, int forecast_index) {
	//predicted_vector->predicted_price->open = date_time_vector[];
}

void predicted_vector_destroy(t_predicted_vector *predicted_vector) {
	free(predicted_vector->predicted_price);
}