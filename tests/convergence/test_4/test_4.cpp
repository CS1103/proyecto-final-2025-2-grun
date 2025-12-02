//
// XOR BCELoss Sigmoid Alt Order Test
//
#include "catch.hpp"
#include "redirect_io.h"
#include "../../../include/utec/algebra/tensor.h"
#include "../../../include/utec/nn/nn_dense.h"
#include "../../../include/utec/nn/nn_activation.h"
#include "../../../include/utec/nn/neural_network.h"
#include <ranges>
#include <random>

static void test_4() {

    using namespace utec::neural_network;

    // Test: XOR problem
    constexpr size_t batch_size = 4;
    Tensor<double,2> X(batch_size, 2);
    Tensor<double,2> Y(batch_size, 1);

    // Datos XOR
    X = {
        0, 1,
        0, 0,
        1, 0,
        1, 1};
    Y = { 1, 0, 1, 0};

    // Inicializador He
    std::mt19937 gen(4);
    auto init_he = [&](Tensor<double,2>& M) {
        const double last = 2.0/(static_cast<double>(M.shape()[0]+ M.shape()[1]));
        std::normal_distribution<double> dist(
            0.0,
            std::sqrt(last));
        for (auto& v : M) v = dist(gen);
    };

    // Construcción de la red
    NeuralNetwork<double> net;
    net.add_layer(std::make_unique<Dense<double>>(
        size_t{2}, size_t{4}, init_he, init_he));
    net.add_layer(std::make_unique<Sigmoid<double>>());
    net.add_layer(std::make_unique<Dense<double>>(
        size_t{4}, size_t{1}, init_he, init_he));
    net.add_layer(std::make_unique<Sigmoid<double>>());

    // Entrenamiento
    constexpr size_t epochs = 4000;
    constexpr double lr = 0.08;
    net.train<BCELoss>(X, Y, epochs, batch_size, lr);

    // Predicción
    Tensor<double,2> Y_prediction = net.predict(X);

    // Verificación
    for (size_t i = 0; i < batch_size; ++i) {
        const double p = Y_prediction(i, 0);
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

TEST_CASE("XOR BCELoss Sigmoid Alt Order") {
    execute_test("test_4.in", test_4);
}
