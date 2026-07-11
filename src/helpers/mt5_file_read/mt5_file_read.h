//
// Created by User on 21.06.2026.
//

#ifndef LSTM_C_PROJECT_MT5_FILE_READ_H
#define LSTM_C_PROJECT_MT5_FILE_READ_H
#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include "../../nn/constants/constants.h"

#ifdef _WIN32
#define FILE_SEPARATOR "\\"
#else
#define FILE_SEPARATOR "/"
#endif

#define MT5_DELIMITER "\t"
#define MT5_DELIMITER_DATE "."
#define MT5_DELIMITER_TIME ":"

typedef struct MT5File t_mt5file;
typedef struct MT5Line t_mt5line;

struct MT5File {
	FILE *file;
	int linesCount;
	t_mt5line *lines;
	char tmp_char;
	char path[PATH_MAX];
};

struct MT5Line {
	bool has_not_error;
	char *buffer;
	int day;
	int month;
	int year;
	int hour;
	int minute;
	double open;
	double high;
	double low;
	double close;
	double volume;
	double *normalize_nn_full_buffer;
	double *normalize_nn_short_buffer;
	double *full_buffer_diff;
	double *short_buffer_diff;
};

void mt5_file_init(t_mt5file *mt5file, char *filename);
void mt5_file_print_normalize_array(t_mt5line *mt5line);
void mt5_file_print_unormalize_array_from_vector(double *vector);
void mt5_file_destroy(t_mt5file *mt5file);

#endif //LSTM_C_PROJECT_MT5_FILE_READ_H
