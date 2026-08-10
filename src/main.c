#include <math.h>
#include <stdio.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "helpers/mt5_file_read/mt5_file_read.h"
#include "helpers/main_struct/main_struct.h"
#include "helpers/predicted_vector/predicted_vector.h"
#include "helpers/weight_factors_helper/weight_factors.h"
#include "nn/lstm/lstm_neural_network/lstm_neural_network.h"

#define DAYS 30
#define HOURS 24
#define CELL_COUNT (24*DAYS)
#define CELL_COUNT_TEST (24*DAYS)
#define STEP_FORECAST 0
#define STEP_FORECAST_TEST 0
#define FORECAST_GAP 1
#define FORECATS_GAP_TEST_MODE 1

t_main_struct *main_struct;
t_lstm_neural_network *lstm_network;
t_lstm_neural_network *lstm_network_first_pointer;
t_lstm_neural_network *lstm_network_last_pointer;
t_predicted_vector *predicted_vector;

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
    main_struct->test_mode = 0;
    main_struct->layers_count = 1;
    main_struct->step_forecasts = STEP_FORECAST;
    main_struct->forecasts_gap = FORECAST_GAP;
    main_struct->cell_count = CELL_COUNT;

    for (int arg_index = 0; arg_index < argc; arg_index++) {
        if (strcmp(argv[arg_index], "-ts") == 0 || strcmp(argv[arg_index], "--trainingsource") == 0) {
            main_struct->training_source_file_path = argv[arg_index + 1];
        }
        if (strcmp(argv[arg_index], "--test") == 0) {
            main_struct->cell_count = CELL_COUNT_TEST;
            main_struct->step_forecasts = STEP_FORECAST_TEST;
            main_struct->forecasts_gap = FORECATS_GAP_TEST_MODE;
            main_struct->test_mode = 1;
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
            main_struct->layers_count = atoi(argv[arg_index + 1]);
        }
        if (strcmp(argv[arg_index], "-cc") == 0 || strcmp(argv[arg_index], "--cellcount") == 0) {
            main_struct->cell_count = atof(argv[arg_index + 1]);
        }
        if (strcmp(argv[arg_index], "-sf") == 0 || strcmp(argv[arg_index], "--stepforecast") == 0) {
            main_struct->step_forecasts = atoi(argv[arg_index + 1]);
        }
        if (strcmp(argv[arg_index], "-fg") == 0 || strcmp(argv[arg_index], "--forecastgap") == 0) {
            main_struct->forecasts_gap = atoi(argv[arg_index + 1]);
        }
    }

    signal(SIGINT, &exit_handler);
    srand(time(NULL));
    lstm_network = malloc(sizeof(t_lstm_neural_network));
    if (main_struct->step_forecasts > 0) {
        lstm_neural_network_init_with_empty_input_vector(lstm_network, main_struct->cell_count, PRICE_BUFFER_SIZE, PREDICT_VECTOR_SIZE, main_struct->step_forecasts);
    } else {
        lstm_neural_network_init(lstm_network, main_struct->cell_count, PRICE_BUFFER_SIZE, PREDICT_VECTOR_SIZE);
    }

    lstm_network->index = 0;
    lstm_network_first_pointer = lstm_network;
    lstm_network_last_pointer = lstm_network;

    if (main_struct->layers_count > 1) {
        for (int layer_index = 1; layer_index < main_struct->layers_count; layer_index++) {
            lstm_network->next = malloc(sizeof(t_lstm_neural_network));
            lstm_neural_network_init(lstm_network->next, main_struct->cell_count + main_struct->step_forecasts, PREDICT_VECTOR_SIZE, PREDICT_VECTOR_SIZE);
            lstm_network->next->index = layer_index;
            lstm_network->next->prev = lstm_network;
            lstm_network = lstm_network->next;
            lstm_network_last_pointer = lstm_network;
        }
        lstm_network = lstm_network_first_pointer;
    }

    t_mt5file *file = malloc(sizeof(t_mt5file));
    if (main_struct->training_source_file_path != 0 && main_struct->test_mode == 1) {
        /*
         * Test mode
         */
        //TODO
        printf("\nTest mode:\n");
        mt5_file_init(file, main_struct->training_source_file_path);
        if (main_struct->weight_factors_file_path != 0) {
            //weight_factors_load_from_file(lstm_network, main_struct);
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

        int file_row_pointer = 1000;
        int file_finish_row_pointer = 90000;
        for (int row_index = file_row_pointer; row_index < file_finish_row_pointer; row_index++) {
            for (int cell_index = 0; cell_index < main_struct->cell_count; cell_index++) {
                lstm_cell_set_inputs(&lstm_network->lstm_cells[cell_index], file->lines[row_index + cell_index].normalize_nn_full_buffer);
            }
            for (int cell_index = 0; cell_index < (main_struct->cell_count + main_struct->step_forecasts); cell_index++) {
                lstm_cell_set_expected_vector(&lstm_network_last_pointer->lstm_cells[cell_index], file->lines[row_index + cell_index + main_struct->forecasts_gap].short_buffer_diff);
            }
            for (int batch_index = 0; batch_index < 10000; batch_index++) {
                lstm_neural_network_learning_step_bptt(lstm_network);
                lstm_neural_network_forward_propagation(lstm_network);
                lstm_neural_network_full_mean_squared_error_calculation(lstm_network);
                printf("expected vector\n");
                for (int cell_index = 0; cell_index < lstm_network_last_pointer->cells_count_full; cell_index++) {
                    lstm_cell_print_any_vector(lstm_network_last_pointer->lstm_cells[cell_index].expected_outputs, lstm_network_last_pointer->lstm_cells[cell_index].node_count_per_single_gate);
                }
                printf("hidden vector\n");
                for (int cell_index = 0; cell_index < lstm_network_last_pointer->cells_count_full; cell_index++) {
                    lstm_cell_print_any_vector(lstm_network_last_pointer->lstm_cells[cell_index].hidden_state, lstm_network_last_pointer->lstm_cells[cell_index].node_count_per_single_gate);
                }
                printf("MSE %3.15f\n", lstm_network_last_pointer->full_mean_squared_error);
                getchar();
            }
        }
    } else if (main_struct->training_source_file_path != 0 && main_struct->test_mode == 0) {
        /* Training mode
         * acceptable mean square error MSE 0.0000000002
         */
        printf("\nTraining mode:\n");
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

        int file_row_pointer = 1000;
        int file_finish_row_pointer = 25000;
        for (int row_index = file_row_pointer; row_index < file_finish_row_pointer; row_index++) {
            for (int cell_index = 0; cell_index < main_struct->cell_count; cell_index++) {
                lstm_cell_set_inputs(&lstm_network->lstm_cells[cell_index], file->lines[row_index + cell_index].normalize_nn_full_buffer);
            }
            for (int cell_index = 0; cell_index < (main_struct->cell_count + main_struct->step_forecasts); cell_index++) {
                lstm_cell_set_expected_vector(&lstm_network_last_pointer->lstm_cells[cell_index], file->lines[row_index + cell_index + main_struct->forecasts_gap].short_buffer_diff);
            }
            for (int batch_index = 0; batch_index < 50; batch_index++) {
                lstm_neural_network_learning_step_bptt(lstm_network);
                lstm_neural_network_forward_propagation(lstm_network);
            }
            lstm_neural_network_full_mean_squared_error_calculation(lstm_network);
            printf("file pointer %d\n", row_index);
            printf("MSE %3.15f\n", lstm_network_last_pointer->full_mean_squared_error);
            if (main_struct->training_source_file_path != 0 && main_struct->weight_factors_file_path != 0) {
                printf("\nSave weight factors to %s\n", main_struct->weight_factors_file_path);
                weight_factors_save_to_file(lstm_network, main_struct);
            }
        }
    } else if (main_struct->source_to_forecast_file_path != 0) {
        /*
         * Forecast mode
         */
        printf("\nForecast mode:\n");
        print_main_struct_info(main_struct);
        mt5_file_init(file, main_struct->source_to_forecast_file_path);
        if (main_struct->weight_factors_file_path != 0) {
            printf("Weight factors file: %s\n", main_struct->weight_factors_file_path);
            weight_factors_load_from_file(lstm_network, main_struct);
        } else {
            printf("Error: file with weights must be defined! (use -w <FILE> or --weight <FILE>)\n");
            return 0;
        }

        predicted_vector = malloc(sizeof(t_predicted_vector));
        predicted_vector_init(predicted_vector, main_struct->step_forecasts);
        predicted_price_init(&predicted_vector->predicted_price[0], &file->lines[main_struct->forecast_from_line -1]);

        printf("\n");
        int start_row = main_struct->forecast_from_line - main_struct->cell_count;
        int final_cell_index_before_predict = main_struct->cell_count - 1;
        printf("start line from file %d\n", start_row);
        for (int cell_index = 0; cell_index < main_struct->cell_count; cell_index++) {
            lstm_cell_set_inputs(&lstm_network->lstm_cells[cell_index], file->lines[start_row + cell_index].normalize_nn_full_buffer);
            if (cell_index == main_struct->cell_count - 1) {

            }
        }
        lstm_neural_network_forward_propagation(lstm_network);
        printf("cell index %d, line in file %d\n[%3.15f %3.15f %3.15f %3.15f]\n", final_cell_index_before_predict, start_row + final_cell_index_before_predict, file->lines[start_row + final_cell_index_before_predict].open, file->lines[start_row + final_cell_index_before_predict].high, file->lines[start_row + final_cell_index_before_predict].low, file->lines[start_row + final_cell_index_before_predict].close);
        predicted_vector_get_data_from_lstm_net(predicted_vector, lstm_network_first_pointer);
        predicted_vector_print(predicted_vector);
        predicted_vector_destroy(predicted_vector);
    }
    lstm_neural_network_destroy(lstm_network);
    free(predicted_vector);
    mt5_file_destroy(file);
    free(lstm_network);
    free(file);
    free(main_struct);
    return 0;
}
