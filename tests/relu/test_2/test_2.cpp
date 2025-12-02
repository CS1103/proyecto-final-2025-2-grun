//
// ReLU Diagonal Pattern Test
//
#include "catch.hpp"
#include "redirect_io.h"
#include "../../../include/utec/algebra/tensor.h"
#include "../../../include/utec/nn/nn_activation.h"
#include <array>
#include <numeric>

template<typename T, size_t DIMS>
using Tensor = utec::algebra::Tensor<T, DIMS>;

using T = float;

template <size_t sz, typename T>
std::array<T, sz> get_array(T initial) {
    std::array<T, sz> result{};
    std::iota(result.begin(), result.end(), initial);
    return result;
}

static void test_2() {
    auto relu = utec::neural_network::ReLU<T>();
    // Tensores
    constexpr int rows = 5;
    constexpr int cols = 4;
    Tensor<T, 2> M(rows, cols);
    M.fill(-0.1f);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) {
            if (i == j) M(i, j) = 4.5;
            if (i == rows - 1 - j) M(i, j) = 4.5;
        }
    std::cout << std::fixed << std::setprecision(1);
    std::cout << M << std::endl;
    // Forward
    const auto R = relu.forward(M);
    const auto clean = apply(R,
        [](auto v){ return (v == T{0}) ? T{0} : v; });
    std::cout << clean << std::endl;
    // Backward
    Tensor<T, 2> GR(rows,cols); GR.fill(1.0f);
    const auto dM = relu.backward(GR);
    std::cout << dM << std::endl;
}

TEST_CASE("ReLU Diagonal Pattern") {
    execute_test("test_2.in", test_2);
}
