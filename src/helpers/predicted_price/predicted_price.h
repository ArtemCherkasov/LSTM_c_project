//
// Created by User on 11.07.2026.
//

#ifndef LSTM_C_PROJECT_PREDICTED_DATA_H
#define LSTM_C_PROJECT_PREDICTED_DATA_H

#include "../../helpers/mt5_file_read/mt5_file_read.h"

typedef struct PredictedPrice t_predicted_price;

struct PredictedPrice {
	double open_output;
	double high_output;
	double low_output;
	double close_output;
	double open;
	double high;
	double low;
	double close;
	double volume;
	int year;
	int month;
	int day;
	int hour;
	int minute;
};

void predicted_price_init(t_predicted_price *predicted_price, t_mt5line *mt5_line);

#endif //LSTM_C_PROJECT_PREDICTED_DATA_H
