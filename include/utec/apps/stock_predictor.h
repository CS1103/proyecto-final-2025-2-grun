#ifndef UTEC_APPS_STOCK_PREDICTOR_H
#define UTEC_APPS_STOCK_PREDICTOR_H

#include "../algebra/tensor.h"
#include "../nn/neural_network.h"
#include "data_loader.h"
#include <vector>
#include <string>
#include <memory>

namespace utec::apps {

// Estructura para almacenar características técnicas de una ventana de tiempo
struct TechnicalFeatures {
    double price_change_1d;      // Cambio de precio 1 día
    double price_change_3d;      // Cambio de precio 3 días
    double price_change_5d;      // Cambio de precio 5 días
    double sma_5;                // Media móvil simple 5 días
    double sma_10;               // Media móvil simple 10 días
    double sma_20;               // Media móvil simple 20 días
    double rsi;                  // Índice de fuerza relativa
    double volume_ratio;         // Ratio volumen actual vs promedio
    double volatility;           // Volatilidad histórica 10 días
    double momentum;             // Momentum (rate of change) 10 días
    
    std::vector<double> to_vector() const;
    static size_t feature_count();
};

// Estructura para una muestra de entrenamiento
struct TrainingSample {
    TechnicalFeatures features;
    int label;  // 1 = sube, 0 = baja
};

class StockPredictor {
private:
    std::unique_ptr<utec::neural_network::NeuralNetwork<double>> model_;
    size_t lookback_days_;
    size_t prediction_days_;
    bool is_trained_;
    
    // Parámetros para normalización
    std::vector<double> feature_means_;
    std::vector<double> feature_stds_;
    
public:
    StockPredictor(size_t lookback_days = 20, size_t prediction_days = 1);
    
    // Configurar la arquitectura de la red neuronal
    void build_model(const std::vector<size_t>& hidden_layers = {64, 32, 16});
    
    // Extraer características técnicas de los datos históricos
    std::vector<TechnicalFeatures> extract_features(const MarketData& data);
    
    // Generar labels binarios (1 = sube, 0 = baja)
    std::vector<int> generate_labels(const MarketData& data);
    
    // Preparar datos de entrenamiento con ventanas deslizantes
    std::vector<TrainingSample> prepare_training_data(const MarketData& data);
    
    // Entrenar el modelo
    void train(const std::vector<MarketData>& training_data, 
               size_t epochs = 100, 
               double learning_rate = 0.001,
               size_t batch_size = 32);
    
    // Hacer predicción binaria (probabilidad de subida)
    double predict_probability(const TechnicalFeatures& features);
    
    // Hacer predicción binaria (0 o 1)
    int predict_binary(const TechnicalFeatures& features, double threshold = 0.5);
    
    // Predecir usando los últimos datos disponibles
    double predict_next(const MarketData& data);
    
    // Evaluar el modelo con datos de prueba
    double evaluate(const std::vector<MarketData>& test_data);
    
    // Guardar y cargar modelo (simplificado)
    void save_model(const std::string& filename);
    void load_model(const std::string& filename);
    
    // Acceso al modelo (para uso en validación/entrenamiento externo)
    utec::neural_network::NeuralNetwork<double>* get_model() { return model_.get(); }
    
    // Marcar modelo como entrenado (para validación externa)
    void set_trained(bool trained) { is_trained_ = trained; }
    
    // Hacer públicos los métodos de normalización y conversión para el validador
    void normalize_features(std::vector<TechnicalFeatures>& features);
    utec::algebra::Tensor<double, 2> features_to_tensor(const std::vector<TechnicalFeatures>& features);
    utec::algebra::Tensor<double, 2> labels_to_tensor(const std::vector<int>& labels);

private:
    // Funciones auxiliares para cálculos técnicos
    double calculate_sma(const std::vector<double>& prices, size_t period, size_t end_idx);
    double calculate_rsi(const std::vector<double>& prices, size_t period, size_t end_idx);
    double calculate_volatility(const std::vector<double>& prices, size_t period, size_t end_idx);
    double calculate_momentum(const std::vector<double>& prices, size_t period, size_t end_idx);
    double calculate_volume_ratio(const std::vector<double>& volumes, size_t period, size_t end_idx);
    
    // Normalización de características individuales
    TechnicalFeatures normalize_single_feature(const TechnicalFeatures& features);
};

} // namespace utec::apps

#endif // UTEC_APPS_STOCK_PREDICTOR_H