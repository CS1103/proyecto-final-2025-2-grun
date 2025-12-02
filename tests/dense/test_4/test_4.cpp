//
// Dense Xavier Initialization Test
//
#include "catch.hpp"
#include "redirect_io.h"
#include "../../../include/utec/nn/nn_dense.h"
#include <random>

static void test_4() {
    using namespace utec::neural_network;
    using T = double;

    // Inicializador Xavier
    std::mt19937 gen(4);
    auto xavier_init = [&](auto& parameter) {
        const double limit = std::sqrt(6.0 / (parameter.shape()[0] + parameter.shape()[1]));
        std::uniform_real_distribution<> dist(-limit, limit);
        for (auto& v : parameter) v = dist(gen);
    };


    constexpr int n_batches = 2;
    constexpr int in_features = 4;
    constexpr int out_features = 3;
    Dense<double> layer(size_t{in_features}, size_t{out_features},xavier_init, xavier_init);

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

TEST_CASE("Dense Xavier Initialization") {
    execute_test("test_4.in", test_4);
}
