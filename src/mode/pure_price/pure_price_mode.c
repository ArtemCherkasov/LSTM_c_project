//
// Created by User on 07.10.2026.
//

#include "pure_price_mode.h"

void print_pure_price_to_standard_output(t_main_struct *main_struct) {
	mt5_file_init(main_struct->file, main_struct->source_file);
	for (int line_index = main_struct->get_from_line; line_index < main_struct->get_from_line + main_struct->get_count; line_index++) {
		printf("%1.5f %1.5f %1.5f %1.5f %1.5f\n", main_struct->file->lines[line_index].open, main_struct->file->lines[line_index].high, main_struct->file->lines[line_index].low, main_struct->file->lines[line_index].close, main_struct->file->lines[line_index].volume);
	}
}