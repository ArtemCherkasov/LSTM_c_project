//
// Created by User on 21.06.2026.
//

#include "mt5_file_read.h"
#include <stdlib.h>
#include <unistd.h>

void mt5_file_init(t_mt5file *mt5file, char *filename) {
	char *token_price;
	char date_buffer[16];
	char time_buffer[16];
	char strtok_buffer[1024];
	char mt5_delimiter[] = {MT5_DELIMITER};
	char mt5_delimiter_date[] = {MT5_DELIMITER_DATE};
	char mt5_delimiter_time[] = {MT5_DELIMITER_TIME};

	mt5file->linesCount = 0;
	getcwd(mt5file->path, sizeof(mt5file->path));
	strcat(mt5file->path, filename);
	mt5file->file = fopen(mt5file->path, "r");
	if (mt5file->file == NULL) {
		printf("\nCould not open file %s\n", mt5file->path);
	}

	for (mt5file->tmp_char = getc(mt5file->file); mt5file->tmp_char != EOF; mt5file->tmp_char = getc(mt5file->file)) {
		if (mt5file->tmp_char == '\n') {
			mt5file->linesCount++;
		}
	}
	rewind(mt5file->file);
	mt5file->lines = malloc(sizeof(t_mt5line) * (mt5file->linesCount + 1));
	for (int line_index = 0; line_index < mt5file->linesCount; line_index++) {
		fgets(mt5file->lines[line_index].buffer, sizeof(mt5file->lines[line_index].buffer), mt5file->file);
		strcpy(strtok_buffer, mt5file->lines[line_index].buffer + sizeof(char));
		/*
		 * parse prices level - open, high, low and close,
		 * also parse volume
		 */
		token_price = strtok(strtok_buffer, mt5_delimiter);
		strcpy(date_buffer, token_price);
		token_price = strtok(NULL, mt5_delimiter);
		strcpy(time_buffer, token_price);
		token_price = strtok(NULL, mt5_delimiter);
		mt5file->lines[line_index].open = atof(token_price);
		mt5file->lines[line_index].normalize_nn_buffer[OPEN_INDEX] = atof(token_price);
		token_price = strtok(NULL, mt5_delimiter);
		mt5file->lines[line_index].high = atof(token_price);
		mt5file->lines[line_index].normalize_nn_buffer[HIGH_INDEX] = atof(token_price);
		token_price = strtok(NULL, mt5_delimiter);
		mt5file->lines[line_index].low = atof(token_price);
		mt5file->lines[line_index].normalize_nn_buffer[LOW_INDEX] = atof(token_price);
		token_price = strtok(NULL, mt5_delimiter);
		mt5file->lines[line_index].close = atof(token_price);
		mt5file->lines[line_index].normalize_nn_buffer[CLOSE_INDEX] = atof(token_price);
		token_price = strtok(NULL, mt5_delimiter);
		mt5file->lines[line_index].volume = atof(token_price);
		mt5file->lines[line_index].normalize_nn_buffer[VOLUME_INDEX] = atof(token_price) / NORMALIZE_FACTOR_VOLUME;
		/*
		 * parse date
		 */
		token_price = strtok(date_buffer, mt5_delimiter_date);
		mt5file->lines[line_index].year = atoi(token_price);
		mt5file->lines[line_index].normalize_nn_buffer[YEAR_INDEX] = atof(token_price) / NORMALIZE_FACTOR_YEAR;
		token_price = strtok(NULL, mt5_delimiter_date);
		mt5file->lines[line_index].month = atoi(token_price);
		mt5file->lines[line_index].normalize_nn_buffer[MONTH_INDEX] = atof(token_price);
		token_price = strtok(NULL, mt5_delimiter_date);
		mt5file->lines[line_index].day = atoi(token_price);
		mt5file->lines[line_index].normalize_nn_buffer[DAY_INDEX] = atof(token_price);
		/*
		 * parse time
		 */
		token_price = strtok(time_buffer, mt5_delimiter_time);
		mt5file->lines[line_index].hour = atoi(token_price);
		mt5file->lines[line_index].normalize_nn_buffer[HOUR_INDEX] = atof(token_price);
		token_price = strtok(NULL, mt5_delimiter_time);
		mt5file->lines[line_index].minute = atoi(token_price);
		mt5file->lines[line_index].normalize_nn_buffer[MINUTE_INDEX] = atof(token_price);

		if (mt5file->lines[line_index].open == 0
		    && mt5file->lines[line_index].high == 0
		    && mt5file->lines[line_index].low == 0
		    && mt5file->lines[line_index].close == 0
		    && mt5file->lines[line_index].volume == 0) {
			mt5file->lines[line_index].has_not_error = false;
		}
	}
	fclose(mt5file->file);
}

void mt5_file_print_normalize_array(t_mt5line *mt5line) {
	for (int index = 0; index < NORMALIZE_NN_BUFFER_SIZE; index++) {
		printf("%3.4f ", mt5line->normalize_nn_buffer[index]);
	}
	printf("\n");
}

void mt5_file_destroy(t_mt5file *mt5file) {
	free(mt5file->lines);
}
