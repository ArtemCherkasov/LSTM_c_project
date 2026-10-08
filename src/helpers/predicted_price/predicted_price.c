//
// Created by User on 11.07.2026.
//

#include "predicted_price.h"

#include "../format_output/format_output.h"

void predicted_price_init(t_predicted_price *predicted_price, t_main_struct *main_struct) {
	predicted_price->open = main_struct->file->lines[main_struct->forecast_from_line - 1].open;
	predicted_price->high = main_struct->file->lines[main_struct->forecast_from_line - 1].high;
	predicted_price->low = main_struct->file->lines[main_struct->forecast_from_line - 1].low;
	predicted_price->close = main_struct->file->lines[main_struct->forecast_from_line - 1].close;
	std_output_line(main_struct->verbose,"%3.15f %3.15f %3.15f %3.15f\n", predicted_price->open, predicted_price->high, predicted_price->low, predicted_price->close);
}
