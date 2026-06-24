//
// Created by User on 08.06.2026.
//

#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "../src/helpers/file_read/mt5_file_read.h"
char path[1024];
char parent_dir[] = "..";
char resource_dir[] = "resources";
char mt5_file_name[] = "EURUSD_H1_200906120000_202509251100.csv";
char file_separator[] = {FILE_SEPARATOR};
double open_1 = 1.40954;
double high_1 = 1.41129;
double low_1 = 1.40929;
double close_1 = 1.41069;
double volume_1 = 1002.0;
int year_101053 = 2025;
double year_101053_norm = 2.025;
int month_101053 = 9;
int day_101053 = 25;
int hour_101053 = 11;
double open_101053 = 1.17332;
double high_101053 = 1.17415;
double low_101053 = 1.17304;
double close_101053 = 1.17367;
double volume_101053 = 1740.0;
double volume_101053_norm = 1.74;

void assert_file_read(char *file_name) {
    t_mt5file *file = malloc(sizeof(t_mt5file));
    mt5_file_init(file, file_name);

    assert(false == file->lines[0].has_not_error);
    assert(open_1 == file->lines[1].open);
    assert(high_1 == file->lines[1].high);
    assert(low_1 == file->lines[1].low);
    assert(close_1 == file->lines[1].close);
    assert(volume_1 == file->lines[1].volume);
    assert(open_101053 == file->lines[101053].open);
    assert(high_101053 == file->lines[101053].high);
    assert(low_101053 == file->lines[101053].low);
    assert(close_101053 == file->lines[101053].close);
    assert(volume_101053 == file->lines[101053].volume);
    assert(year_101053 == file->lines[101053].year);
    assert(month_101053 == file->lines[101053].month);
    assert(day_101053 == file->lines[101053].day);
    assert(hour_101053 == file->lines[101053].hour);
    assert(open_101053 == file->lines[101053].normalize_nn_buffer[OPEN_INDEX]);
    assert(high_101053 == file->lines[101053].normalize_nn_buffer[HIGH_INDEX]);
    assert(low_101053 == file->lines[101053].normalize_nn_buffer[LOW_INDEX]);
    assert(close_101053 == file->lines[101053].normalize_nn_buffer[CLOSE_INDEX]);
    assert(volume_101053_norm == file->lines[101053].normalize_nn_buffer[VOLUME_INDEX]);
    assert(year_101053_norm == file->lines[101053].normalize_nn_buffer[YEAR_INDEX]);
    assert(month_101053 == file->lines[101053].normalize_nn_buffer[MONTH_INDEX]);
    assert(day_101053 == file->lines[101053].normalize_nn_buffer[DAY_INDEX]);
    assert(hour_101053 == file->lines[101053].normalize_nn_buffer[HOUR_INDEX]);

    for (int line_index = 0; line_index < file->linesCount; line_index++) {
        mt5_file_print_normalize_array(&file->lines[line_index]);
    }
    mt5_file_destroy(file);
    free(file);
}

int main() {
    strcat(path, file_separator);
    strcat(path, parent_dir);
    strcat(path, file_separator);
    strcat(path, parent_dir);
    strcat(path, file_separator);
    strcat(path, resource_dir);
    strcat(path, file_separator);
    strcat(path, mt5_file_name);
    printf("%s\n", path);
    assert_file_read(path);
    return 0;
}
