//
// Created by User on 09.06.2026.
//

#ifndef LSTM_C_PROJECT_CONSTANTS_H
#define LSTM_C_PROJECT_CONSTANTS_H

extern const double BIAS_DEFAULT_VALUE;
extern const int BIASES_COUNT;
extern const double WEIGHT_DELTA_VALUE;
extern const double EPS;
extern const int INPUT_COUNT_PER_CELL;
extern const int INPUT_COUNT_PER_TAIL_CELL;
extern const int PREDICT_VECTOR_SIZE;
extern const int DAY_INDEX;
extern const int MONTH_INDEX;
extern const int YEAR_INDEX; // deprecated
extern const int WEEKDAY_INDEX;
extern const int HOUR_INDEX;
extern const int MINUTE_INDEX;
extern const int OPEN_INDEX;
extern const int HIGH_INDEX;
extern const int LOW_INDEX;
extern const int CLOSE_INDEX;
extern const int VOLUME_INDEX;
extern const int SHORT_OPEN_INDEX;
extern const int SHORT_HIGH_INDEX;
extern const int SHORT_LOW_INDEX;
extern const int SHORT_CLOSE_INDEX;
extern const int SHORT_VOLUME_INDEX;
extern const int HOUR_SHIFT;
extern const int MINUTE_SHIFT;
extern const double NORMALIZE_FACTOR_YEAR;
extern const double NORMALIZE_FACTOR_WEEK_DAY;
extern const double NORMALIZE_FACTOR_MONTH;
extern const double NORMALIZE_FACTOR_DAY;
extern const double NORMALIZE_FACTOR_VOLUME;
extern const double NORMALIZE_FACTOR_PRICE;
extern const double NORMALIZE_FACTOR_HOUR;
extern const double NORMALIZE_FACTOR_MINUTE;
extern const double NORMALIZE_FACTOR_DIFF;
extern const double HIDDEN_STATE_FACTOR;
extern const double CORRECTION_TO_SIGMA_MIDDLE;
extern const double AMOUNT_OF_EXPANSION_SIGMA;
extern const int DAYS;
extern const int HOURS;
extern const int CELL_COUNT;
extern const int CELL_COUNT_TEST;
extern const int STEP_FORECAST;
extern const int STEP_FORECAST_TEST;
extern const int FORECAST_GAP;
extern const int FORECATS_GAP_TEST_MODE;
#endif //LSTM_C_PROJECT_CONSTANTS_H
