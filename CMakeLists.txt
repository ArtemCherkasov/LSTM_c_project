cmake_minimum_required(VERSION 4.2)
project(LSTM_c_project C)

set(CMAKE_C_STANDARD 11)
enable_testing()
add_executable(LSTM_c_project src/main.c
        src/nn/common/layer/layer.h
        src/nn/common/layer/layer.c
        src/nn/common/node/node.h
        src/nn/common/node/node.c
        src/nn/common/bias/bias.c
        src/nn/common/bias/bias.h
        src/nn/constants/constants.h
        src/nn/constants/constants.c)

add_library(node_library src/nn/common/node/node.c)
add_library(layer_library src/nn/common/layer/layer.c)
add_library(bias_library src/nn/common/bias/bias.c)
add_library(constants_library src/nn/constants/constants.c)
add_subdirectory(test)
