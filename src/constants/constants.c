//
// Created by User on 15.06.2026.
//
#include "constants.h"

const double BIAS_DEFAULT_VALUE = 1.000;
const int BIASES_COUNT = 1;
const double WEIGHT_DELTA_VALUE = 0.015;
const int INPUT_COUNT_PER_CELL = 10;
const int INPUT_COUNT_PER_TAIL_CELL = 5;
const int PREDICT_VECTOR_SIZE = 4;
const int DAY_INDEX = 0;
const int MONTH_INDEX = 1;
const int YEAR_INDEX = 2;
const int WEEKDAY_INDEX = 2;
const int HOUR_INDEX = 3;
const int MINUTE_INDEX = 4;
const int OPEN_INDEX = 5;
const int HIGH_INDEX = 6;
const int LOW_INDEX = 7;
const int CLOSE_INDEX = 8;
const int VOLUME_INDEX = 9;
const int SHORT_OPEN_INDEX = 0;
const int SHORT_HIGH_INDEX = 1;
const int SHORT_LOW_INDEX = 2;
const int SHORT_CLOSE_INDEX = 3;
const int SHORT_VOLUME_INDEX = 4;
const int HOUR_SHIFT = 1;
const int MINUTE_SHIFT = 1;
const double NORMALIZE_FACTOR_YEAR = 10000.0;
const double NORMALIZE_FACTOR_WEEK_DAY = 10.0;
const double NORMALIZE_FACTOR_MONTH = 1000.0;
const double NORMALIZE_FACTOR_DAY = 10000.0;
const double NORMALIZE_FACTOR_VOLUME = 10000.0;
const double NORMALIZE_FACTOR_PRICE = 10.0;
const double NORMALIZE_FACTOR_HOUR = 100.0;
const double NORMALIZE_FACTOR_MINUTE = 100.0;
const double NORMALIZE_FACTOR_DIFF = 65.0;
const double HIDDEN_STATE_FACTOR = 1.0;
const double CORRECTION_TO_SIGMA_MIDDLE = 0.5;
const double AMOUNT_OF_EXPANSION_SIGMA = 1200.0;
const double EPS = 1e-15;
const int DAYS = 30;
const int HOURS = 24;
const int CELL_COUNT = (24*DAYS);
const int CELL_COUNT_TEST = (24*DAYS);
const int STEP_FORECAST = 0;
const int STEP_FORECAST_TEST = 0;
const int FORECAST_GAP = 1;
const int FORECATS_GAP_TEST_MODE = 1;

