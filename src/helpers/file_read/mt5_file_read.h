//
// Created by User on 21.06.2026.
//

#ifndef LSTM_C_PROJECT_MT5_FILE_READ_H
#define LSTM_C_PROJECT_MT5_FILE_READ_H
#include <limits.h>
#include <stdbool.h>
#include <stdio.h>

#ifdef _WIN32
#define FILE_SEPARATOR '\\'
#else
#define FILE_SEPARATOR '/'
#endif

#define MT5_DELIMITER "\t"
#define MT5_DELIMITER_DATE "."
#define MT5_DELIMITER_TIME ":"

#define NORMALIZE_NN_BUFFER_SIZE 10
#define DAY_INDEX 0
#define MONTH_INDEX 1
#define YEAR_INDEX 2
#define HOUR_INDEX 3
#define MINUTE_INDEX 4
#define OPEN_INDEX 5
#define HIGH_INDEX 6
#define LOW_INDEX 7
#define CLOSE_INDEX 8
#define VOLUME_INDEX 9
#define NORMALIZE_FACTOR_YEAR 1000.0
#define NORMALIZE_FACTOR_VOLUME 1000.0

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
	char buffer[1024];
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
	double normalize_nn_buffer[NORMALIZE_NN_BUFFER_SIZE];
};

void mt5_file_init(t_mt5file *mt5file, char *filename);
void mt5_file_print_normalize_array(t_mt5line *mt5line);
void mt5_file_destroy(t_mt5file *mt5file);

#endif //LSTM_C_PROJECT_MT5_FILE_READ_H
