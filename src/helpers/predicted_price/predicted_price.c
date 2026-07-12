//
// Created by User on 11.07.2026.
//

#include "predicted_price.h"

void predicted_price_init(t_predicted_price *predicted_price, t_mt5line *mt5_line) {
	predicted_price->open = mt5_line->open;
	predicted_price->high = mt5_line->high;
	predicted_price->low = mt5_line->low;
	predicted_price->close = mt5_line->close;
	printf("%3.15f %3.15f %3.15f %3.15f\n", predicted_price->open, predicted_price->high, predicted_price->low, predicted_price->close);
}