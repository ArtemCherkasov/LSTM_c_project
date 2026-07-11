#include <stdio.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "helpers/mt5_file_read/mt5_file_read.h"
#include "helpers/main_struct/main_struct.h"
#include "helpers/weight_factors_helper/weight_factors.h"
#include "nn/lstm/lstm_neural_network/lstm_neural_network.h"

#define DAYS 20
#define CELL_COUNT (24*DAYS)
#define STEP_FORECAST 1

t_main_struct *main_struct;
t_lstm_neural_network *lstm_network;
t_lstm_neural_network *lstm_network_first_pointer;
t_lstm_neural_network *lstm_network_last_pointer;

void exit_handler(int n_signal) {
    printf("\nCODE %d\n", n_signal);
    if (main_struct->training_source_file_path != 0 && main_struct->weight_factors_file_path != 0) {
        printf("\nSave weight factors to %s\n", main_struct->weight_factors_file_path);
        weight_factors_save_to_file(lstm_network, main_struct);
    }
    printf("\nExiting ...\n");
    raise(SIGTERM);
}

int main(int argc, char *argv[]) {

    main_struct = malloc(sizeof(t_main_struct));
    main_struct->training_source_file_path = 0;
    main_struct->source_to_forecast_file_path = 0;
    main_struct->weight_factors_file_path = 0;
    main_struct->price_symbol = 0;
    main_struct->learning_rate = 0;
    main_struct->layers_count = 1;
    main_struct->step_forecasts = STEP_FORECAST;

    for (int arg_index = 0; arg_index < argc; arg_index++) {
        if (strcmp(argv[arg_index], "-ts") == 0 || strcmp(argv[arg_index], "--trainingsource") == 0) {
            main_struct->training_source_file_path = argv[arg_index + 1];
        }
        if (strcmp(argv[arg_index], "-fs") == 0 || strcmp(argv[arg_index], "--forecastsource") == 0) {
            main_struct->source_to_forecast_file_path = argv[arg_index + 1];
        }
        if (strcmp(argv[arg_index], "-ffl") == 0 || strcmp(argv[arg_index], "--forecastfromline") == 0) {
            main_struct->forecast_from_line = atoi(argv[arg_index + 1]);
        }
        if (strcmp(argv[arg_index], "-s") == 0 || strcmp(argv[arg_index], "--symbol") == 0) {
            main_struct->price_symbol = argv[arg_index + 1];
        }
        if (strcmp(argv[arg_index], "-w") == 0 || strcmp(argv[arg_index], "--weight") == 0) {
            main_struct->weight_factors_file_path = argv[arg_index + 1];
        }
        if (strcmp(argv[arg_index], "-lr") == 0 || strcmp(argv[arg_index], "--learningrate") == 0) {
            main_struct->learning_rate = atof(argv[arg_index + 1]);
        }
        if (strcmp(argv[arg_index], "-lc") == 0 || strcmp(argv[arg_index], "--layerscount") == 0) {
            main_struct->layers_count = atof(argv[arg_index + 1]);
        }
        if (strcmp(argv[arg_index], "-sf") == 0 || strcmp(argv[arg_index], "--stepforecast") == 0) {
            main_struct->step_forecasts = atoi(argv[arg_index + 1]);
        }
    }

    signal(SIGINT, &exit_handler);
    lstm_network = malloc(sizeof(t_lstm_neural_network));
    lstm_neural_network_init(lstm_network, CELL_COUNT, PRICE_BUFFER_SIZE, PREDICT_VECTOR_SIZE);
    lstm_network->index = 0;
    lstm_network_first_pointer = lstm_network;

    if (main_struct->layers_count > 1) {
        for (int layer_index = 1; layer_index < main_struct->layers_count; layer_index++) {
            lstm_network->next = malloc(sizeof(t_lstm_neural_network));
            lstm_neural_network_init(lstm_network->next, CELL_COUNT, PREDICT_VECTOR_SIZE, PREDICT_VECTOR_SIZE);
            lstm_network->next->index = layer_index;
            lstm_network = lstm_network->next;
            lstm_network_last_pointer = lstm_network;
        }
        lstm_network = lstm_network_first_pointer;
    }

    t_mt5file *file = malloc(sizeof(t_mt5file));
    if (main_struct->training_source_file_path != 0) {
        /*
         * acceptable mean square error MSE 0.0000000002
         */
        printf("\nTraining mode:\n");
        srand(time(NULL));
        mt5_file_init(file, main_struct->training_source_file_path);
        if (main_struct->weight_factors_file_path != 0) {
            weight_factors_load_from_file(lstm_network, main_struct);
        }
        if (main_struct->learning_rate != 0) {
            printf("Set learning rate: %0.5f\n", main_struct->learning_rate);
            lstm_network->learning_rate = main_struct->learning_rate;
            if (main_struct->layers_count > 1) {
                for (int layer_index = 1; layer_index < main_struct->layers_count; layer_index++) {
                    lstm_network->next->learning_rate = main_struct->learning_rate;
                    lstm_network = lstm_network->next;
                }
                lstm_network = lstm_network_first_pointer;
            }
        }

        int file_row_pointer = 100;
        int file_finish_row_pointer = 10000;
        for (int row_index = file_row_pointer; row_index < file_finish_row_pointer; row_index++) {
            for (int cell_index = 0; cell_index < CELL_COUNT; cell_index++) {
                lstm_cell_set_inputs(&lstm_network->lstm_cells[cell_index], file->lines[row_index + cell_index].full_buffer_diff);
                lstm_cell_set_expected_vector(&lstm_network_last_pointer->lstm_cells[cell_index], file->lines[main_struct->step_forecasts + row_index + cell_index].short_buffer_diff);
            }
            lstm_neural_network_learning_step(lstm_network);
            lstm_neural_network_forward_propagation(lstm_network);
            lstm_neural_network_full_mean_squared_error_calculation(lstm_network);
            printf("MSE %3.15f\n", lstm_network_last_pointer->full_mean_squared_error);
            if (main_struct->training_source_file_path != 0 && main_struct->weight_factors_file_path != 0) {
                printf("\nSave weight factors to %s\n", main_struct->weight_factors_file_path);
                weight_factors_save_to_file(lstm_network, main_struct);
            }
        }
    } else if (main_struct->source_to_forecast_file_path != 0) {
        printf("\nForecast mode:\n");
        mt5_file_init(file, main_struct->source_to_forecast_file_path);
        if (main_struct->weight_factors_file_path != 0) {
            printf("Weight factors file: %s\n", main_struct->weight_factors_file_path);
            weight_factors_load_from_file(lstm_network, main_struct);
        } else {
            printf("Error: file with weights must be defined! (use -w <FILE> or --weight <FILE>)\n");
            return 0;
        }
        int start_row = main_struct->forecast_from_line - CELL_COUNT;
        for (int cell_index = 0; cell_index < CELL_COUNT; cell_index++) {
            lstm_cell_set_inputs(&lstm_network->lstm_cells[cell_index], file->lines[start_row + cell_index].normalize_nn_full_buffer);
        }
        lstm_neural_network_forward_propagation(lstm_network);
        for (int cell_index = CELL_COUNT - STEP_FORECAST; cell_index < CELL_COUNT; cell_index++) {
            mt5_file_print_unormalize_array_from_vector(lstm_network_last_pointer->lstm_cells[cell_index].hidden_state);
        }
    }
    lstm_neural_network_destroy(lstm_network);
    mt5_file_destroy(file);
    free(lstm_network);
    free(file);
    free(main_struct);
    return 0;
}
