add_subdirectory(external/rl-tools)
add_executable(bindings_test l2f.cpp)
target_compile_definitions(bindings_test PRIVATE TEST)
target_link_libraries(bindings_test PRIVATE rl_tools)
target_compile_options(bindings_test PRIVATE -Ofast)

