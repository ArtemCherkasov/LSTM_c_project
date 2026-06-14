add_executable(node_test node_test.c)
target_link_libraries(node_test node_library)
target_link_libraries(node_test constants_library)
add_test(NAME NodeTests COMMAND node_test)