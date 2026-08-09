//
// Created by User on 11.07.2026.
//

#ifndef LSTM_C_PROJECT_PREDICRED_VECTOR_H
#define LSTM_C_PROJECT_PREDICRED_VECTOR_H

#include "../../nn/lstm/lstm_neural_network/lstm_neural_network.h"
#include "../predicted_price/predicted_price.h"

typedef struct PredictedVector t_predicted_vector;

struct PredictedVector {
	t_predicted_price *predicted_price;
	int predicted_vector_size;
};

void predicted_vector_init(t_predicted_vector *predicted_vector, int forecast_gap);
void predicted_vector_get_data_from_lstm_net(t_predicted_vector *predicted_vector, t_lstm_neural_network *lstm_network);
void predicted_vector_print(t_predicted_vector *predicted_vector);
void predicted_vector_destroy(t_predicted_vector *predicted_vector);

#endif //LSTM_C_PROJECT_PREDICRED_VECTOR_H
