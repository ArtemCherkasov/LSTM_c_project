//
// Created by User on 15.06.2026.
//
#include "constants.h"

const double BIAS_DEFAULT_VALUE = 1.000;
const int BIASES_COUNT = 1;
const double WEIGHT_DELTA_VALUE = 0.015;
const int PRICE_BUFFER_SIZE = 10;
const int PREDICT_VECTOR_SIZE = 4;
const int DAY_INDEX = 0;
const int MONTH_INDEX = 1;
const int YEAR_INDEX = 2;
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
const double NORMALIZE_FACTOR_YEAR = 10000.0;
const double NORMALIZE_FACTOR_MONTH = 100.0;
const double NORMALIZE_FACTOR_DAY = 100.0;
const double NORMALIZE_FACTOR_VOLUME = 10000.0;
const double NORMALIZE_FACTOR_PRICE = 10.0;
const double NORMALIZE_FACTOR_HOUR = 100.0;
const double NORMALIZE_FACTOR_MINUTE = 100.0;
const double NORMALIZE_FACTOR_DIFF = 100.0;
const double HIDDEN_STATE_FACTOR = 1.0;
const double CORRECTION_TO_SIGMA_MIDDLE = 0.5;
const double EPS = 1e-15;