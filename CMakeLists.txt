cmake_minimum_required(VERSION 3.20)
project(neural_network CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra)
endif()

find_package(BLAS REQUIRED)
find_package(GTest REQUIRED)

option(WITH_DML "Enable DirectML GPU Acceleration" OFF)

include_directories(include)

set(NEURAL_NETWORK_SOURCES
    src/core/tensor.cpp
    src/backend/cpu_backend.cpp
    src/nn/layer.cpp
    src/nn/loss.cpp
    src/optim/optimizer.cpp
    src/optim/lr_scheduler.cpp
    src/optim/dataloader.cpp
    src/optim/trainer.cpp
    src/core/serialization.cpp
    src/backend/thread_pool.cpp
    src/optim/grad_scaler.cpp
    src/nn/lora.cpp
)

if(WITH_DML)
    add_compile_definitions(WITH_DML)
    list(APPEND NEURAL_NETWORK_SOURCES src/backend/dml_backend.cpp)
else()
    list(APPEND NEURAL_NETWORK_SOURCES src/backend/dml_backend_stub.cpp)
endif()

add_library(neural_network STATIC ${NEURAL_NETWORK_SOURCES})

if(WITH_DML)
    target_link_libraries(neural_network PUBLIC ${BLAS_LIBRARIES} d3d12 dxgi DirectML)
else()
    target_link_libraries(neural_network PUBLIC ${BLAS_LIBRARIES})
endif()

add_executable(neural_network_main src/main.cpp)
target_link_libraries(neural_network_main PRIVATE neural_network)

enable_testing()

add_executable(tensor_test tests/core/tensor_test.cpp)
target_link_libraries(tensor_test PRIVATE neural_network GTest::gtest GTest::gtest_main)
add_test(NAME TensorTest COMMAND tensor_test)

add_executable(layer_test tests/nn/layer_test.cpp)
target_link_libraries(layer_test PRIVATE neural_network GTest::gtest GTest::gtest_main)
add_test(NAME LayerTest COMMAND layer_test)

add_executable(optimizer_test tests/optim/optimizer_test.cpp)
target_link_libraries(optimizer_test PRIVATE neural_network GTest::gtest GTest::gtest_main)
add_test(NAME OptimizerTest COMMAND optimizer_test)

add_executable(pipeline_test tests/optim/pipeline_test.cpp)
target_link_libraries(pipeline_test PRIVATE neural_network GTest::gtest GTest::gtest_main)
add_test(NAME PipelineTest COMMAND pipeline_test)

add_executable(serialization_test tests/core/serialization_test.cpp)
target_link_libraries(serialization_test PRIVATE neural_network GTest::gtest GTest::gtest_main)
add_test(NAME SerializationTest COMMAND serialization_test)

add_executable(transformer_test tests/nn/transformer_test.cpp)
target_link_libraries(transformer_test PRIVATE neural_network GTest::gtest GTest::gtest_main)
add_test(NAME TransformerTest COMMAND transformer_test)

add_executable(optimizer_advanced_test tests/optim/optimizer_advanced_test.cpp)
target_link_libraries(optimizer_advanced_test PRIVATE neural_network GTest::gtest GTest::gtest_main)
add_test(NAME OptimizerAdvancedTest COMMAND optimizer_advanced_test)

add_executable(dml_test tests/backend/dml_test.cpp)
target_link_libraries(dml_test PRIVATE neural_network GTest::gtest GTest::gtest_main)
add_test(NAME DmlTest COMMAND dml_test)

add_executable(rope_swiglu_test tests/nn/rope_swiglu_test.cpp)
target_link_libraries(rope_swiglu_test PRIVATE neural_network GTest::gtest GTest::gtest_main)
add_test(NAME RopeSwigluTest COMMAND rope_swiglu_test)

add_executable(parallel_test tests/backend/parallel_test.cpp)
target_link_libraries(parallel_test PRIVATE neural_network GTest::gtest GTest::gtest_main)
add_test(NAME ParallelTest COMMAND parallel_test)

add_executable(amp_test tests/optim/amp_test.cpp)
target_link_libraries(amp_test PRIVATE neural_network GTest::gtest GTest::gtest_main)
add_test(NAME AMPTest COMMAND amp_test)

add_executable(lora_test tests/nn/lora_test.cpp)
target_link_libraries(lora_test PRIVATE neural_network GTest::gtest GTest::gtest_main)
add_test(NAME LoRATest COMMAND lora_test)
