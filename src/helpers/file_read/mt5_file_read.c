//
// Created by User on 21.06.2026.
//

#include "mt5_file_read.h"
#include <stdlib.h>
#include <unistd.h>

void mt5_file_init(t_mt5file *mt5file, char *filename) {
	char *token;
	char strtok_buffer[1024];
	char mt5_delimiter[] = {MT5_DELIMITER};

	mt5file->linesCount = 0;
	getcwd(mt5file->path, sizeof(mt5file->path));
	strcat(mt5file->path, filename);
	mt5file->file = fopen(mt5file->path, "r");
	printf("%s\n", mt5file->path);

	for (mt5file->tmp_char = getc(mt5file->file); mt5file->tmp_char != EOF; mt5file->tmp_char = getc(mt5file->file)) {
		if (mt5file->tmp_char == '\n') {
			mt5file->linesCount++;
		}
	}
	rewind(mt5file->file);
	mt5file->lines = malloc(sizeof(t_mt5line) * (mt5file->linesCount + 1));
	for (int line_index = 0; line_index < mt5file->linesCount; line_index++) {
		fgets(mt5file->lines[line_index].buffer, sizeof(mt5file->lines[line_index].buffer), mt5file->file);
		strcpy(strtok_buffer, mt5file->lines[line_index].buffer);
		token = strtok(strtok_buffer, mt5_delimiter);
		token = strtok(NULL, mt5_delimiter);
		token = strtok(NULL, mt5_delimiter);
		mt5file->lines[line_index].open = atof(token);
		token = strtok(NULL, mt5_delimiter);
		mt5file->lines[line_index].high = atof(token);
		token = strtok(NULL, mt5_delimiter);
		mt5file->lines[line_index].low = atof(token);
		token = strtok(NULL, mt5_delimiter);
		mt5file->lines[line_index].close = atof(token);
		token = strtok(NULL, mt5_delimiter);
		mt5file->lines[line_index].volume = atof(token);
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

void mt5_file_destroy(t_mt5file *mt5file) {
	free(mt5file->lines);
}