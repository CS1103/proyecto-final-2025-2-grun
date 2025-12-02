//
// Dense Forward Identity Init Test
//
#include "catch.hpp"
#include "redirect_io.h"
#include "../../../include/utec/nn/nn_dense.h"

static void test_1() {
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
    auto init_zero = [](Tensor<T,2>& M) { for (auto& v : M) v = 0.0; };
    constexpr int n_batches = 2;
    constexpr int in_features = 3;
    constexpr int out_features = 3;

    Dense<double> layer(size_t{in_features}, size_t{in_features},init_identity, init_zero);

    Tensor<T,2> X(n_batches, in_features);
    X = {
        1.0,  2.0,  3.0,
        -1.5, 4.2,  0.0
    };

    // Forward
    Tensor<T,2> Y = layer.forward(X);

    // Formato
    REQUIRE(Y.shape()[0] == n_batches);
    REQUIRE(Y.shape()[1] == out_features);

    // Y == X elementwise
    for (size_t i = 0; i < n_batches; ++i)
        for (size_t j = 0; j < out_features; ++j)
            REQUIRE(Y(i,j) == Approx(X(i,j)).epsilon(1e-12));
}

TEST_CASE("Dense Forward Identity Init") {
    execute_test("test_1.in", test_1);
}
