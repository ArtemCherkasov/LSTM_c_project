//
// Created by User on 08.06.2026.
//

#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include "../src/helpers/mt5_file_read/mt5_file_read.h"

#define EPS 1e-9

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
int month_101053 = 9;
int day_101053 = 25;
int hour_101053 = 11;
double open_101053 = 1.17332;
double high_101053 = 1.17415;
double low_101053 = 1.17304;
double close_101053 = 1.17367;
double volume_101053 = 1740.0;

double year_101053_norm = 0.202500;
double month_101053_norm = 0.09;
double day_101053_norm = 0.25;
double hour_101053_norm = 0.11;
double open_101053_norm = 0.117332;
double open_101053_norm_2 = 0.117332;
double high_101053_norm = 0.117415;
double low_101053_norm = 0.117304;
double close_101053_norm = 0.117367;
double volume_101053_norm = 0.174000;

void assert_file_read(char *file_name) {
    t_mt5file *file = malloc(sizeof(t_mt5file));
    mt5_file_init(file, file_name);

    assert(false == file->lines[0].has_not_error);
    assert(fabs(open_1 - file->lines[1].open) < EPS);
    assert(fabs(high_1 - file->lines[1].high) < EPS);
    assert(fabs(low_1 - file->lines[1].low) < EPS);
    assert(fabs(close_1 - file->lines[1].close) < EPS);
    assert(fabs(volume_1 - file->lines[1].volume) < EPS);
    assert(fabs(open_101053 - file->lines[101053].open) < EPS);
    assert(fabs(high_101053 - file->lines[101053].high) < EPS);
    assert(fabs(low_101053 - file->lines[101053].low) < EPS);
    assert(fabs(close_101053 - file->lines[101053].close) < EPS);
    assert(fabs(volume_101053 - file->lines[101053].volume) < EPS);
    assert(year_101053 == file->lines[101053].year);
    assert(month_101053 == file->lines[101053].month);
    assert(day_101053 == file->lines[101053].day);
    assert(hour_101053 == file->lines[101053].hour);

    assert(fabs(open_101053_norm - file->lines[101053].normalize_nn_full_buffer[OPEN_INDEX]) < EPS);
    assert(fabs(high_101053_norm - file->lines[101053].normalize_nn_full_buffer[HIGH_INDEX]) < EPS);
    assert(fabs(low_101053_norm - file->lines[101053].normalize_nn_full_buffer[LOW_INDEX]) < EPS);
    assert(fabs(close_101053_norm - file->lines[101053].normalize_nn_full_buffer[CLOSE_INDEX]) < EPS);
    assert(fabs(volume_101053_norm - file->lines[101053].normalize_nn_full_buffer[VOLUME_INDEX]) < EPS);
    assert(fabs(year_101053_norm - file->lines[101053].normalize_nn_full_buffer[YEAR_INDEX]) < EPS);
    assert(fabs(month_101053_norm - file->lines[101053].normalize_nn_full_buffer[MONTH_INDEX]) < EPS);
    assert(fabs(day_101053_norm - file->lines[101053].normalize_nn_full_buffer[DAY_INDEX]) < EPS);
    assert(fabs(hour_101053_norm - file->lines[101053].normalize_nn_full_buffer[HOUR_INDEX]) < EPS);

    mt5_file_destroy(file);
    free(file);
}

int main() {
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
