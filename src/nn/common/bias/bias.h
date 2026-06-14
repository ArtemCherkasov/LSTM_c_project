//
// Created by User on 08.06.2026.
//

#ifndef LSTM_C_PROJECT_BIAS_H
#define LSTM_C_PROJECT_BIAS_H
#include "../../constants/constants.h"
typedef struct Bias t_bias;

struct Bias {
    double biasValue;
};

void bias_init(t_bias *bias);
#endif //LSTM_C_PROJECT_BIAS_H
