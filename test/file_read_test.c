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
double open_101053 = 1.17332;
double high_101053 = 1.17415;
double low_101053 = 1.17304;
double close_101053 = 1.17367;
double volume_101053 = 1740.0;

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
