//
// ReLU Forward-Backward Simple Test
//
#include "catch.hpp"
#include "redirect_io.h"
#include "../../../include/utec/algebra/tensor.h"
#include "../../../include/utec/nn/nn_activation.h"

template<typename T, size_t DIMS>
using Tensor = utec::algebra::Tensor<T, DIMS>;

static void test_1() {
    using T = float;
    auto relu = utec::neural_network::ReLU<T>();
    // Tensores
    Tensor<T, 2> M(2,2); M = {-1, 2, 0, -3};
    Tensor<T, 2> GR(2,2); GR.fill(1.0f);
    // Forward
    auto R = relu.forward(M);
    std::cout << R(0,1) << "\n"; // espera 2
    // Backward
    const auto dM = relu.backward(GR);
    std::cout << dM;
}

TEST_CASE("ReLU Forward-Backward Simple") {
    execute_test("test_1.in", test_1);
}
