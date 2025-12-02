//
// Dense He Initialization Test
//
#include "catch.hpp"
#include "redirect_io.h"
#include "../../../include/utec/nn/nn_dense.h"
#include <random>

static void test_3() {
    using namespace utec::neural_network;
    using T = double;

    // Inicializador He
    std::mt19937 gen(42);
    auto init_he = [&](Tensor<double,2>& M) {
        const double last = 2.0/(static_cast<double>(M.shape()[0]+ M.shape()[1]));
        std::normal_distribution<double> dist(
            0.0,
            std::sqrt(last));
        for (auto& v : M) v = dist(gen);
    };

    constexpr int n_batches = 2;
    constexpr int in_features = 4;
    constexpr int out_features = 3;
    Dense<double> layer(size_t{in_features}, size_t{out_features},init_he, init_he);

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


TEST_CASE("Dense He Initialization") {
    execute_test("test_3.in", test_3);
}
