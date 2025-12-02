//
// Dense Backward Iota Test
//
#include "catch.hpp"
#include "redirect_io.h"
#include "../../../include/utec/nn/nn_dense.h"

static void test_2() {
    using namespace utec::neural_network;
    using T = double;

    // Inicializador identidad
    auto init_identity = [](Tensor<T,2>& M) {
        const auto shape = M.shape();
        const size_t rows = shape[0];
        const size_t cols = shape[1];
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                M(i,j) = (i == j ? 1.0 : 0.0);
    };

    // Inicializador de ceros
    auto init_zero = [](Tensor<T,2>& M) {
        for (auto& v : M) v = 0.0;
    };
    constexpr int n_batches = 2;
    constexpr int in_features = 4;
    constexpr int out_features = 3;
    Dense<double> layer(size_t{in_features}, size_t{out_features},init_identity, init_zero);

    Tensor<T,2> X1(n_batches, in_features);
    std::iota(X1.begin(), X1.end(), 1);
    // Forward
    Tensor<T,2> Y = layer.forward(X1);
    std::cout << Y << std::endl;

    Tensor<T,2> Z(n_batches, out_features);
    std::iota(Z.begin(), Z.end(), 1);
    auto Z_adjusted = Z / static_cast<T>(Z.size());

    Tensor<T,2> X_adjusted = layer.backward(Z_adjusted);
    // X ajustado
    std::cout << X_adjusted << std::endl;
}

TEST_CASE("Dense Backward Iota") {
    execute_test("test_2.in", test_2);
}
