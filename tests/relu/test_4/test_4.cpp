//
// ReLU Gradient Validation Test
//
#include "catch.hpp"
#include "redirect_io.h"
#include "../../../include/utec/algebra/tensor.h"
#include "../../../include/utec/nn/nn_activation.h"

template<typename T, size_t DIMS>
using Tensor = utec::algebra::Tensor<T, DIMS>;

using T = float;

static void test_4() {
    auto relu = utec::neural_network::ReLU<T>();
    // Tensores
    Tensor<T, 2> M(3,3); 
    M = {-2.5f, 0.0f, 3.5f,
         1.2f, -0.8f, 2.1f,
         -1.0f, 4.0f, -0.5f};
    
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "Input:\n" << M << std::endl;
    
    // Forward
    auto R = relu.forward(M);
    std::cout << "Output:\n" << R << std::endl;
    
    // Backward
    Tensor<T, 2> GR(3,3); GR.fill(1.0f);
    const auto dM = relu.backward(GR);
    std::cout << "Gradient:\n" << dM << std::endl;
}

TEST_CASE("ReLU Gradient Validation") {
    execute_test("test_4.in", test_4);
}
