//
// Created by User on 11.07.2026.
//

#ifndef LSTM_C_PROJECT_PREDICTED_DATA_H
#define LSTM_C_PROJECT_PREDICTED_DATA_H

typedef struct PredictedPrice t_predicted_price;

struct PredictedPrice {
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

#endif //LSTM_C_PROJECT_PREDICTED_DATA_H
