//
// XOR MSELoss ReLU Low LR Test
//
#include "catch.hpp"
#include "redirect_io.h"
#include "../../../include/utec/algebra/tensor.h"
#include "../../../include/utec/nn/nn_dense.h"
#include "../../../include/utec/nn/nn_activation.h"
#include "../../../include/utec/nn/neural_network.h"
#include <ranges>
#include <random>

using namespace std;

static void test_3() {

    using namespace utec::neural_network;

    // Test: XOR problem
    constexpr size_t batch_size = 4;
    Tensor<double,2> X(batch_size, 2);
    Tensor<double,2> Y(batch_size, 1);

    // Datos XOR
    X = { 0, 0,
          0, 1,
          1, 0,
          1, 1};
    Y = { 0, 1, 1, 0};

    // Inicializador Xavier
    std::mt19937 gen(20);
    auto xavier_init = [&gen](auto& parameter) {
        const double limit = std::sqrt(6.0 / (parameter.shape()[0] + parameter.shape()[1]));
        std::uniform_real_distribution<> dist(-limit, limit);
        for (auto& v : parameter) v = dist(gen);
    };

    // Construcción de la red
    NeuralNetwork<double> net;
    net.add_layer(std::make_unique<Dense<double>>(
        size_t{2}, size_t{4}, xavier_init, xavier_init));
    net.add_layer(std::make_unique<ReLU<double>>());
    net.add_layer(std::make_unique<Dense<double>>(
        size_t{4}, size_t{1}, xavier_init, xavier_init));

    // Entrenamiento
    constexpr size_t epochs = 4000;
    constexpr double learning_rate = 0.02;
    net.train<MSELoss> (X, Y, epochs, batch_size, learning_rate);

    // Predicción
    Tensor<double,2> Y_prediction = net.predict(X);

    // Verificación
    for (size_t i = 0; i < batch_size; ++i) {
        const double p = Y_prediction(i,0);
        std::cout
            << std::fixed << std::setprecision(0)
            << "Input: (" << X(i,0) << "," << X(i,1)
            << std::fixed << std::setprecision(4)
            << ") -> Prediction: " << p << std::endl;
        if (Y(i,0) < 0.5) {
            REQUIRE(p < 0.5); // Expected output close to 0
        } else {
            REQUIRE(p >= 0.6); // Expected output close to 1
        }
    }
}

TEST_CASE("XOR MSELoss ReLU Low LR") {
    execute_test("test_3.in", test_3);
}
