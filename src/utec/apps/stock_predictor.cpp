#include "../../../include/utec/apps/stock_predictor.h"
#include "../../../include/utec/nn/nn_dense.h"
#include "../../../include/utec/nn/nn_activation.h"
#include "../../../include/utec/nn/nn_loss.h"
#include "../../../include/utec/nn/nn_optimizer.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <random>

namespace utec::apps {

// TechnicalFeatures implementation
std::vector<double> TechnicalFeatures::to_vector() const {
    return {price_change_1d, price_change_3d, price_change_5d, 
            sma_5, sma_10, sma_20, rsi, volume_ratio, 
            volatility, momentum};
}

size_t TechnicalFeatures::feature_count() {
    return 10; // número de características
}

// StockPredictor implementation
StockPredictor::StockPredictor(size_t lookback_days, size_t prediction_days)
    : lookback_days_(lookback_days), prediction_days_(prediction_days), is_trained_(false) {
    model_ = std::make_unique<utec::neural_network::NeuralNetwork<double>>();
}

void StockPredictor::build_model(const std::vector<size_t>& hidden_layers) {
    using namespace utec::neural_network;
    
    size_t input_size = TechnicalFeatures::feature_count();
    
    // Funciones de inicialización simples
    auto init_weights = [](auto& tensor) {
        for (auto& val : tensor) {
            val = (static_cast<double>(rand()) / RAND_MAX) * 0.1 - 0.05; // [-0.05, 0.05]
        }
    };
    
    auto init_bias = [](auto& tensor) {
        for (auto& val : tensor) {
            val = 0.0;
        }
    };
    
    // Primera capa densa
    model_->add_layer(std::make_unique<Dense<double>>(input_size, hidden_layers[0], init_weights, init_bias));
    model_->add_layer(std::make_unique<ReLU<double>>());
    
    // Capas ocultas
    for (size_t i = 1; i < hidden_layers.size(); ++i) {
        model_->add_layer(std::make_unique<Dense<double>>(hidden_layers[i-1], hidden_layers[i], init_weights, init_bias));
        model_->add_layer(std::make_unique<ReLU<double>>());
    }
    
    // Capa de salida (1 neurona para predicción binaria)
    model_->add_layer(std::make_unique<Dense<double>>(hidden_layers.back(), 1, init_weights, init_bias));
    model_->add_layer(std::make_unique<Sigmoid<double>>());
    
    std::cout << "Modelo construido con " << hidden_layers.size() << " capas ocultas" << std::endl;
}

std::vector<TechnicalFeatures> StockPredictor::extract_features(const MarketData& data) {
    std::vector<TechnicalFeatures> features;
    const auto& prices = data.prices;
    const auto& volumes = data.volumes;
    
    // Necesitamos al menos lookback_days_ + 20 días para calcular todas las características
    size_t min_days = std::max(lookback_days_, size_t(25));
    
    for (size_t i = min_days; i < prices.size(); ++i) {
        TechnicalFeatures feature;
        
        // Cambios de precio
        feature.price_change_1d = (prices[i] - prices[i-1]) / prices[i-1];
        feature.price_change_3d = (prices[i] - prices[i-3]) / prices[i-3];
        feature.price_change_5d = (prices[i] - prices[i-5]) / prices[i-5];
        
        // Medias móviles
        feature.sma_5 = calculate_sma(prices, 5, i);
        feature.sma_10 = calculate_sma(prices, 10, i);
        feature.sma_20 = calculate_sma(prices, 20, i);
        
        // Indicadores técnicos
        feature.rsi = calculate_rsi(prices, 14, i);
        feature.volatility = calculate_volatility(prices, 10, i);
        feature.momentum = calculate_momentum(prices, 10, i);
        feature.volume_ratio = calculate_volume_ratio(volumes, 10, i);
        
        features.push_back(feature);
    }
    
    return features;
}

std::vector<int> StockPredictor::generate_labels(const MarketData& data) {
    std::vector<int> labels;
    const auto& prices = data.prices;
    
    // Necesitamos al menos lookback_days_ + 20 días para las características
    size_t min_days = std::max(lookback_days_, size_t(25));
    
    for (size_t i = min_days; i < prices.size() - prediction_days_; ++i) {
        double current_price = prices[i];
        double future_price = prices[i + prediction_days_];
        
        // Label 1 si el precio sube, 0 si baja
        labels.push_back((future_price > current_price) ? 1 : 0);
    }
    
    return labels;
}

std::vector<TrainingSample> StockPredictor::prepare_training_data(const MarketData& data) {
    auto features = extract_features(data);
    auto labels = generate_labels(data);
    
    // Asegurar que tenemos el mismo número de características y labels
    size_t min_size = std::min(features.size(), labels.size());
    features.resize(min_size);
    labels.resize(min_size);
    
    std::vector<TrainingSample> samples;
    for (size_t i = 0; i < min_size; ++i) {
        samples.push_back({features[i], labels[i]});
    }
    
    std::cout << "Preparados " << samples.size() << " muestras de entrenamiento" << std::endl;
    return samples;
}

void StockPredictor::train(const std::vector<MarketData>& training_data, 
                          size_t epochs, double learning_rate, size_t batch_size) {
    
    // Combinar datos de entrenamiento de todas las acciones
    std::vector<TrainingSample> all_samples;
    for (const auto& data : training_data) {
        auto samples = prepare_training_data(data);
        all_samples.insert(all_samples.end(), samples.begin(), samples.end());
    }
    
    if (all_samples.empty()) {
        throw std::runtime_error("No hay muestras de entrenamiento disponibles");
    }
    
    // Separar características y labels
    std::vector<TechnicalFeatures> features;
    std::vector<int> labels;
    
    for (const auto& sample : all_samples) {
        features.push_back(sample.features);
        labels.push_back(sample.label);
    }
    
    // Normalizar características
    normalize_features(features);
    
    // Convertir a tensores
    auto X = features_to_tensor(features);
    auto Y = labels_to_tensor(labels);
    
    std::cout << "Entrenando con " << X.shape()[0] << " muestras, " 
              << X.shape()[1] << " características" << std::endl;
    
    // Entrenar modelo
    using namespace utec::neural_network;
    model_->template train<BCE, SGD>(X, Y, epochs, batch_size, learning_rate);
    
    is_trained_ = true;
    std::cout << "Entrenamiento completado" << std::endl;
}

double StockPredictor::predict_probability(const TechnicalFeatures& features) {
    if (!is_trained_) {
        throw std::runtime_error("El modelo no ha sido entrenado");
    }
    
    // Normalizar características
    auto normalized = normalize_single_feature(features);
    
    // Convertir a tensor
    auto feature_vec = normalized.to_vector();
    std::array<size_t, 2> tensor_shape = {1, feature_vec.size()};
    utec::algebra::Tensor<double, 2> X(tensor_shape);
    
    for (size_t i = 0; i < feature_vec.size(); ++i) {
        X(0, i) = feature_vec[i];
    }
    
    // Hacer predicción
    auto prediction = model_->predict(X);
    return prediction(0, 0);
}

int StockPredictor::predict_binary(const TechnicalFeatures& features, double threshold) {
    double probability = predict_probability(features);
    return (probability >= threshold) ? 1 : 0;
}

double StockPredictor::predict_next(const MarketData& data) {
    auto features = extract_features(data);
    if (features.empty()) {
        throw std::runtime_error("No se pudieron extraer características de los datos");
    }
    
    // Usar las características más recientes
    return predict_probability(features.back());
}

double StockPredictor::evaluate(const std::vector<MarketData>& test_data) {
    size_t correct_predictions = 0;
    size_t total_predictions = 0;
    
    for (const auto& data : test_data) {
        auto samples = prepare_training_data(data);
        
        for (const auto& sample : samples) {
            int predicted = predict_binary(sample.features);
            if (predicted == sample.label) {
                correct_predictions++;
            }
            total_predictions++;
        }
    }
    
    double accuracy = static_cast<double>(correct_predictions) / total_predictions;
    std::cout << "Precisión: " << accuracy * 100 << "% (" 
              << correct_predictions << "/" << total_predictions << ")" << std::endl;
    
    return accuracy;
}

// Funciones auxiliares para cálculos técnicos
double StockPredictor::calculate_sma(const std::vector<double>& prices, size_t period, size_t end_idx) {
    if (end_idx < period - 1) return prices[end_idx];
    
    double sum = 0.0;
    for (size_t i = end_idx - period + 1; i <= end_idx; ++i) {
        sum += prices[i];
    }
    return sum / period;
}

double StockPredictor::calculate_rsi(const std::vector<double>& prices, size_t period, size_t end_idx) {
    if (end_idx < period) return 50.0; // RSI neutral
    
    double gains = 0.0, losses = 0.0;
    
    for (size_t i = end_idx - period + 1; i <= end_idx; ++i) {
        double change = prices[i] - prices[i-1];
        if (change > 0) gains += change;
        else losses -= change;
    }
    
    double avg_gain = gains / period;
    double avg_loss = losses / period;
    
    if (avg_loss == 0.0) return 100.0;
    
    double rs = avg_gain / avg_loss;
    return 100.0 - (100.0 / (1.0 + rs));
}

double StockPredictor::calculate_volatility(const std::vector<double>& prices, size_t period, size_t end_idx) {
    if (end_idx < period) return 0.0;
    
    // Calcular retornos
    std::vector<double> returns;
    for (size_t i = end_idx - period + 1; i <= end_idx; ++i) {
        returns.push_back((prices[i] - prices[i-1]) / prices[i-1]);
    }
    
    // Calcular media de retornos
    double mean_return = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
    
    // Calcular varianza
    double variance = 0.0;
    for (double ret : returns) {
        variance += (ret - mean_return) * (ret - mean_return);
    }
    variance /= returns.size();
    
    return std::sqrt(variance) * std::sqrt(252); // Volatilidad anualizada
}

double StockPredictor::calculate_momentum(const std::vector<double>& prices, size_t period, size_t end_idx) {
    if (end_idx < period) return 0.0;
    
    double old_price = prices[end_idx - period];
    double current_price = prices[end_idx];
    
    return (current_price - old_price) / old_price;
}

double StockPredictor::calculate_volume_ratio(const std::vector<double>& volumes, size_t period, size_t end_idx) {
    if (end_idx < period - 1) return 1.0;
    
    double avg_volume = 0.0;
    for (size_t i = end_idx - period + 1; i <= end_idx; ++i) {
        avg_volume += volumes[i];
    }
    avg_volume /= period;
    
    return (avg_volume > 0) ? volumes[end_idx] / avg_volume : 1.0;
}

void StockPredictor::normalize_features(std::vector<TechnicalFeatures>& features) {
    if (features.empty()) return;
    
    size_t feature_count = TechnicalFeatures::feature_count();
    feature_means_.resize(feature_count, 0.0);
    feature_stds_.resize(feature_count, 0.0);
    
    // Calcular medias
    for (const auto& feature : features) {
        auto vec = feature.to_vector();
        for (size_t i = 0; i < feature_count; ++i) {
            feature_means_[i] += vec[i];
        }
    }
    
    for (auto& mean : feature_means_) {
        mean /= features.size();
    }
    
    // Calcular desviaciones estándar
    for (const auto& feature : features) {
        auto vec = feature.to_vector();
        for (size_t i = 0; i < feature_count; ++i) {
            double diff = vec[i] - feature_means_[i];
            feature_stds_[i] += diff * diff;
        }
    }
    
    for (auto& std : feature_stds_) {
        std = std::sqrt(std / features.size());
        if (std == 0.0) std = 1.0; // Evitar división por cero
    }
    
    // Normalizar características
    for (auto& feature : features) {
        auto vec = feature.to_vector();
        for (size_t i = 0; i < feature_count; ++i) {
            vec[i] = (vec[i] - feature_means_[i]) / feature_stds_[i];
        }
        
        // Actualizar la estructura
        feature.price_change_1d = vec[0];
        feature.price_change_3d = vec[1];
        feature.price_change_5d = vec[2];
        feature.sma_5 = vec[3];
        feature.sma_10 = vec[4];
        feature.sma_20 = vec[5];
        feature.rsi = vec[6];
        feature.volume_ratio = vec[7];
        feature.volatility = vec[8];
        feature.momentum = vec[9];
    }
}

TechnicalFeatures StockPredictor::normalize_single_feature(const TechnicalFeatures& features) {
    TechnicalFeatures normalized = features;
    auto vec = features.to_vector();
    
    if (feature_means_.empty() || feature_stds_.empty()) {
        return normalized; // No hay parámetros de normalización
    }
    
    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = (vec[i] - feature_means_[i]) / feature_stds_[i];
    }
    
    normalized.price_change_1d = vec[0];
    normalized.price_change_3d = vec[1];
    normalized.price_change_5d = vec[2];
    normalized.sma_5 = vec[3];
    normalized.sma_10 = vec[4];
    normalized.sma_20 = vec[5];
    normalized.rsi = vec[6];
    normalized.volume_ratio = vec[7];
    normalized.volatility = vec[8];
    normalized.momentum = vec[9];
    
    return normalized;
}

utec::algebra::Tensor<double, 2> StockPredictor::features_to_tensor(const std::vector<TechnicalFeatures>& features) {
    std::array<size_t, 2> tensor_shape = {features.size(), TechnicalFeatures::feature_count()};
    utec::algebra::Tensor<double, 2> tensor(tensor_shape);
    
    for (size_t i = 0; i < features.size(); ++i) {
        auto vec = features[i].to_vector();
        for (size_t j = 0; j < vec.size(); ++j) {
            tensor(i, j) = vec[j];
        }
    }
    
    return tensor;
}

utec::algebra::Tensor<double, 2> StockPredictor::labels_to_tensor(const std::vector<int>& labels) {
    std::array<size_t, 2> tensor_shape = {labels.size(), 1};
    utec::algebra::Tensor<double, 2> tensor(tensor_shape);
    
    for (size_t i = 0; i < labels.size(); ++i) {
        tensor(i, 0) = static_cast<double>(labels[i]);
    }
    
    return tensor;
}

// Implementaciones simplificadas para guardar/cargar modelo
void StockPredictor::save_model(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("No se pudo crear el archivo: " + filename);
    }
    
    // Guardar parámetros de normalización
    file << "NORMALIZATION_PARAMS\n";
    file << feature_means_.size() << "\n";
    for (double mean : feature_means_) {
        file << mean << " ";
    }
    file << "\n";
    
    for (double std : feature_stds_) {
        file << std << " ";
    }
    file << "\n";
    
    file << is_trained_ << "\n";
    file << lookback_days_ << "\n";
    file << prediction_days_ << "\n";
    
    std::cout << "Modelo guardado en " << filename << std::endl;
}

void StockPredictor::load_model(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("No se pudo abrir el archivo: " + filename);
    }
    
    std::string header;
    file >> header;
    
    if (header != "NORMALIZATION_PARAMS") {
        throw std::runtime_error("Formato de archivo inválido");
    }
    
    size_t param_count;
    file >> param_count;
    
    feature_means_.resize(param_count);
    feature_stds_.resize(param_count);
    
    for (size_t i = 0; i < param_count; ++i) {
        file >> feature_means_[i];
    }
    
    for (size_t i = 0; i < param_count; ++i) {
        file >> feature_stds_[i];
    }
    
    file >> is_trained_ >> lookback_days_ >> prediction_days_;
    
    std::cout << "Modelo cargado desde " << filename << std::endl;
}

} // namespace utec::apps