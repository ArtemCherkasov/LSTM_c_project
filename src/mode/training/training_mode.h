//
// Created by User on 07.10.2026.
//

#ifndef LSTM_C_PROJECT_TRAINING_MODE_H
#define LSTM_C_PROJECT_TRAINING_MODE_H
#include "../../helpers/main_struct/main_struct.h"
#include "../../helpers/predicted_vector/predicted_vector.h"
#include "../../helpers/weight_factors_helper/weight_factors.h"
#include "../../helpers/mt5_file_read/mt5_file_read.h"

void training_process(t_lstm_neural_network *lstm_network, t_lstm_neural_network *lstm_network_first_pointer, t_lstm_neural_network *lstm_network_last_pointer, t_main_struct *main_struct);
#endif //LSTM_C_PROJECT_TRAINING_MODE_H
