#include <stdio.h>
#include <signal.h>
#include <stdlib.h>
#include <time.h>

#include "helpers/file_read/mt5_file_read.h"
#include "nn/lstm/lstm_neural_network/lstm_neural_network.h"

#define DAYS 10
#define CELL_COUNT (24*DAYS)
#define STEP_FORECAST 24

void exit_handler(int n_signal) {
    printf("\nCODE %d\n", n_signal);
    printf("Exiting ...\n");
}

int main(int argc, char *argv[]) {
    signal(SIGINT, &exit_handler);
    srand(time(NULL));
    t_lstm_neural_network *lstm_network = malloc(sizeof(t_lstm_neural_network));
    t_mt5file *file = malloc(sizeof(t_mt5file));
    mt5_file_init(file, argv[1]);
    lstm_neural_network_init(lstm_network, CELL_COUNT, NORMALIZE_NN_BUFFER_SIZE, NORMALIZE_NN_BUFFER_SIZE);
    int file_row_pointer = 100;
    int file_finish_row_pointer = 10000;



    for (int row_index = file_row_pointer; row_index < file_finish_row_pointer; row_index++) {
        for (int cell_index = 0; cell_index < CELL_COUNT; cell_index++) {
            lstm_cell_set_inputs(&lstm_network->lstm_cells[cell_index], file->lines[row_index + cell_index].normalize_nn_buffer);
            lstm_cell_set_expected_vector(&lstm_network->lstm_cells[cell_index], file->lines[STEP_FORECAST + row_index + cell_index].normalize_nn_buffer);
        }
        lstm_neural_network_learning_step(lstm_network);
        lstm_neural_network_forward_propagation(lstm_network);
        lstm_neural_network_full_mean_squared_error_calculation(lstm_network);
        printf("MSE %3.15f\n", lstm_network->full_mean_squared_error);
    }

    lstm_neural_network_destroy(lstm_network);
    free(lstm_network);
}