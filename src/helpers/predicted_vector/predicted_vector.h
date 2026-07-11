//
// Created by User on 11.07.2026.
//

#ifndef LSTM_C_PROJECT_PREDICRED_VECTOR_H
#define LSTM_C_PROJECT_PREDICRED_VECTOR_H
#include "../predicted_price/predicted_price.h"

typedef struct PredictedVector t_predicted_vector;

struct PredictedVector {
	t_predicted_price *predicted_price;
};

void predicted_vector_init(t_predicted_vector *predicted_vector, int step_forecasts);
void predicted_vector_set_date_time_to_price(t_predicted_vector *predicted_vector, double *date_time_vector, int forecast_index);
void predicted_vector_destroy(t_predicted_vector *predicted_vector);

#endif //LSTM_C_PROJECT_PREDICRED_VECTOR_H
