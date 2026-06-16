add_executable(node_test node_test.c)
add_executable(layer_test layer_test.c)
add_executable(lstm_cell_test lstm_cell_test.c)

target_link_libraries(node_test node_library)
target_link_libraries(node_test constants_library)

target_link_libraries(layer_test layer_library)
target_link_libraries(layer_test node_library)
target_link_libraries(layer_test bias_library)
target_link_libraries(layer_test constants_library)

target_link_libraries(lstm_cell_test lstm_cell_library)
target_link_libraries(lstm_cell_test layer_library)
target_link_libraries(lstm_cell_test node_library)
target_link_libraries(lstm_cell_test bias_library)
target_link_libraries(lstm_cell_test constants_library)

add_test(NAME NodeTests COMMAND node_test)
add_test(NAME LayerTests COMMAND layer_test)
add_test(NAME LSTMCellTests COMMAND lstm_cell_test)