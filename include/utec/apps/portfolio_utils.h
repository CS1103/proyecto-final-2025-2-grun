#ifndef UTEC_APPS_PORTFOLIO_UTILS_H
#define UTEC_APPS_PORTFOLIO_UTILS_H

#include "../nn/nn_dense.h"
#include "../nn/nn_activation.h"
#include "../algebra/tensor.h"
#include <random>
#include <memory>

namespace utec::apps {

using namespace utec::neural_network;
using namespace utec::algebra;

/**
 * @brief Funciones de inicialización para los pesos de la red neuronal
 */
class NetworkInitializers {
public:
    // Inicialización Xavier/Glorot uniforme
    template<typename T>
    static auto xavier_uniform(std::mt19937& rng) {
        return [&rng](utec::algebra::Tensor<T, 2>& tensor) {
            const auto& shape = tensor.shape();
            T fan_in = static_cast<T>(shape[0]);
            T fan_out = static_cast<T>(shape[1]);
            T limit = std::sqrt(T{6.0} / (fan_in + fan_out));
            
            std::uniform_real_distribution<T> dist(-limit, limit);
            for (auto& val : tensor) {
                val = dist(rng);
            }
        };
    }
    
    // Inicialización de sesgos a cero
    template<typename T>
    static auto zeros() {
        return [](utec::algebra::Tensor<T, 2>& tensor) {
            tensor.fill(T{0});
        };
    }
    
    // Inicialización He (para ReLU)
    template<typename T>
    static auto he_normal(std::mt19937& rng) {
        return [&rng](utec::algebra::Tensor<T, 2>& tensor) {
            const auto& shape = tensor.shape();
            T fan_in = static_cast<T>(shape[0]);
            T stddev = std::sqrt(T{2.0} / fan_in);
            
            std::normal_distribution<T> dist(T{0}, stddev);
            for (auto& val : tensor) {
                val = dist(rng);
            }
        };
    }
};

/**
 * @brief Builder para crear redes neuronales fácilmente
 */
template<typename T>
class NetworkBuilder {
private:
    std::unique_ptr<NeuralNetwork<T>> network_;
    std::mt19937 rng_;
    size_t current_size_;
    
public:
    explicit NetworkBuilder(size_t input_size, uint32_t seed = 42) 
        : network_(std::make_unique<NeuralNetwork<T>>()), 
          rng_(seed), 
          current_size_(input_size) {
    }
    
    NetworkBuilder& add_dense(size_t units, bool use_he_init = true) {
        if (use_he_init) {
            auto dense = std::make_unique<Dense<T>>(
                current_size_, units,
                NetworkInitializers::template he_normal<T>(rng_),
                NetworkInitializers::template zeros<T>()
            );
            network_->add_layer(std::move(dense));
        } else {
            auto dense = std::make_unique<Dense<T>>(
                current_size_, units,
                NetworkInitializers::template xavier_uniform<T>(rng_),
                NetworkInitializers::template zeros<T>()
            );
            network_->add_layer(std::move(dense));
        }
        current_size_ = units;
        return *this;
    }
    
    NetworkBuilder& add_relu() {
        auto relu = std::make_unique<ReLU<T>>();
        network_->add_layer(std::move(relu));
        return *this;
    }
    
    NetworkBuilder& add_sigmoid() {
        auto sigmoid = std::make_unique<Sigmoid<T>>();
        network_->add_layer(std::move(sigmoid));
        return *this;
    }
    
    NetworkBuilder& add_softmax() {
        auto softmax = std::make_unique<Softmax<T>>();
        network_->add_layer(std::move(softmax));
        return *this;
    }
    
    std::unique_ptr<NeuralNetwork<T>> build() {
        return std::move(network_);
    }
    
    size_t get_current_size() const { return current_size_; }
};

/**
 * @brief Métricas de evaluación para portfolios
 */
class PortfolioMetrics {
public:
    // Calcular Sharpe Ratio
    static double sharpe_ratio(const std::vector<double>& returns, double risk_free_rate = 0.02) {
        if (returns.size() < 2) return 0.0;
        
        double mean_return = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
        double variance = 0.0;
        for (double ret : returns) {
            variance += (ret - mean_return) * (ret - mean_return);
        }
        variance /= (returns.size() - 1);
        double volatility = std::sqrt(variance);
        
        if (volatility == 0.0) return 0.0;
        
        // Anualizar
        double annualized_return = mean_return * 252;
        double annualized_vol = volatility * std::sqrt(252);
        
        return (annualized_return - risk_free_rate) / annualized_vol;
    }
    
    // Calcular Maximum Drawdown
    static double maximum_drawdown(const std::vector<double>& portfolio_values) {
        if (portfolio_values.size() < 2) return 0.0;
        
        double peak = portfolio_values[0];
        double max_dd = 0.0;
        
        for (double value : portfolio_values) {
            if (value > peak) {
                peak = value;
            }
            double drawdown = (peak - value) / peak;
            max_dd = std::max(max_dd, drawdown);
        }
        
        return max_dd;
    }
    
    // Calcular Value at Risk (VaR) al 95%
    static double value_at_risk(std::vector<double> returns, double confidence = 0.95) {
        if (returns.empty()) return 0.0;
        
        std::sort(returns.begin(), returns.end());
        size_t index = static_cast<size_t>((1.0 - confidence) * returns.size());
        return -returns[index]; // Negativo porque VaR se reporta como pérdida potencial
    }
};

/**
 * @brief Estrategias de benchmark simples
 */
class BenchmarkStrategies {
public:
    // Estrategia Buy & Hold (pesos iguales, sin rebalanceo)
    static std::vector<double> buy_and_hold_weights(size_t num_assets) {
        return std::vector<double>(num_assets, 1.0 / num_assets);
    }
    
    // Estrategia de momentum (mayor peso a activos con mejor rendimiento reciente)
    static std::vector<double> momentum_weights(const std::vector<double>& recent_returns) {
        std::vector<double> weights = recent_returns;
        
        // Convertir a pesos positivos
        double min_return = *std::min_element(weights.begin(), weights.end());
        for (double& w : weights) {
            w = w - min_return + 0.01; // Offset para evitar pesos cero
        }
        
        // Normalizar
        double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
        for (double& w : weights) {
            w /= sum;
        }
        
        return weights;
    }
    
    // Estrategia de volatilidad inversa (mayor peso a activos menos volátiles)
    static std::vector<double> inverse_volatility_weights(const std::vector<double>& volatilities) {
        std::vector<double> weights;
        weights.reserve(volatilities.size());
        
        // Calcular inversos de volatilidad
        for (double vol : volatilities) {
            weights.push_back(1.0 / std::max(vol, 0.01)); // Evitar división por cero
        }
        
        // Normalizar
        double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
        for (double& w : weights) {
            w /= sum;
        }
        
        return weights;
    }
};

} // namespace utec::apps

#endif // UTEC_APPS_PORTFOLIO_UTILS_H