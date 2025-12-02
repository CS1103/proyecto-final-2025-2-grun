//
// Sigmoid Forward-Backward Test
//
#include "catch.hpp"
#include "redirect_io.h"
#include "../../../include/utec/algebra/tensor.h"
#include "../../../include/utec/nn/nn_activation.h"
#include <array>

template<typename T, size_t DIMS>
using Tensor = utec::algebra::Tensor<T, DIMS>;

using T = double;

static void test_3() {
    auto sigmoid = utec::neural_network::Sigmoid<T>();
    // Tensores
    constexpr int rows = 5;
    constexpr int cols = 4;
    Tensor<T, 2> M(rows, cols);
    M.fill(-100.0);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) {
            if (i == j) M(i, j) = 100.0;
            if (i == rows - 1 - j) M(i, j) = 100.0;
        }
    std::cout << std::fixed << std::setprecision(1);
    std::cout << M << std::endl;
    // Forward
    const auto S = sigmoid.forward(M);
    std::cout << S << std::endl;
    // Backward
    Tensor<T, 2> GR(rows,cols); GR.fill(1.0);
    const auto dM = sigmoid.backward(GR);
    std::cout << dM << std::endl;
}

TEST_CASE("Sigmoid Forward-Backward") {
    execute_test("test_3.in", test_3);
}
