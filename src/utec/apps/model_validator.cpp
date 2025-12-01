#include "../../../include/utec/apps/stock_predictor.h"
#include "../../../include/utec/apps/data_loader.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iomanip>
#include <memory>
#include <cmath>

using namespace utec::apps;

struct CSVSample {
    std::string date;
    std::string symbol;
    double close;
    TechnicalFeatures features;
    int actual_target;
};

class SimpleValidator {
private:
    std::vector<CSVSample> training_samples_;
    std::vector<CSVSample> test_samples_;
    std::unique_ptr<StockPredictor> predictor_;
    
public:
    SimpleValidator() {
        // Crear predictor con configuración estándar
        predictor_ = std::make_unique<StockPredictor>(5, 1);
        
        // Construir el modelo de red neuronal
        std::vector<size_t> hidden_layers = {64, 32, 16};
        predictor_->build_model(hidden_layers);
    }
    
    bool load_training_data(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cout << "Error: No se puede abrir " << filename << std::endl;
            return false;
        }
        
        std::string line;
        std::getline(file, line); // Skip header
        
        while (std::getline(file, line)) {
            CSVSample sample;
            if (parse_csv_line(line, sample)) {
                training_samples_.push_back(sample);
            }
        }
        
        std::cout << "Cargadas " << training_samples_.size() << " muestras de entrenamiento" << std::endl;
        return !training_samples_.empty();
    }
    
    bool load_test_data(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cout << "Error: No se puede abrir " << filename << std::endl;
            return false;
        }
        
        std::string line;
        std::getline(file, line); // Skip header
        
        while (std::getline(file, line)) {
            CSVSample sample;
            if (parse_csv_line(line, sample)) {
                test_samples_.push_back(sample);
            }
        }
        
        std::cout << "Cargadas " << test_samples_.size() << " muestras de prueba" << std::endl;
        return !test_samples_.empty();
    }
    
    bool parse_csv_line(const std::string& line, CSVSample& sample) {
        std::stringstream ss(line);
        std::string item;
        std::vector<std::string> tokens;
        
        while (std::getline(ss, item, ',')) {
            tokens.push_back(item);
        }
        
        if (tokens.size() < 13) return false;
        
        try {
            sample.date = tokens[0];
            sample.symbol = tokens[1];
            sample.close = std::stod(tokens[2]);
            
            // Parse features in the same order as TechnicalFeatures::to_vector()
            sample.features.price_change_1d = std::stod(tokens[3]);
            sample.features.price_change_3d = std::stod(tokens[4]);
            sample.features.price_change_5d = std::stod(tokens[5]);
            sample.features.sma_5 = std::stod(tokens[6]);
            sample.features.sma_10 = std::stod(tokens[7]);
            sample.features.sma_20 = std::stod(tokens[8]);
            sample.features.rsi = std::stod(tokens[9]);
            sample.features.volume_ratio = std::stod(tokens[10]);
            sample.features.volatility = std::stod(tokens[11]);
            sample.features.momentum = std::stod(tokens[12]);
            sample.actual_target = std::stoi(tokens[13]);
            
            return true;
        } catch (const std::exception& e) {
            return false;
        }
    }
    
    void train_and_validate() {
        if (training_samples_.empty()) {
            std::cout << "No hay datos de entrenamiento" << std::endl;
            return;
        }
        
        if (test_samples_.empty()) {
            std::cout << "No hay datos de prueba" << std::endl;
            return;
        }
        
        std::cout << "\n=== ENTRENAMIENTO DE RED NEURONAL ===" << std::endl;
        std::cout << "Muestras de entrenamiento: " << training_samples_.size() << std::endl;
        std::cout << "Muestras de prueba: " << test_samples_.size() << std::endl;
        
        // Preparar datos de entrenamiento
        std::vector<TrainingSample> training_data;
        for (const auto& sample : training_samples_) {
            TrainingSample ts;
            ts.features = sample.features;
            ts.label = sample.actual_target;
            training_data.push_back(ts);
        }
        
        // Entrenar el predictor
        std::cout << "\nEntrenando con " << training_data.size() << " muestras..." << std::endl;
        train_predictor(training_data);
        
        // Validar con los datos de prueba
        validate_with_test_data();
    }
    
    void train_predictor(const std::vector<TrainingSample>& samples) {
        // Separar características y etiquetas
        std::vector<TechnicalFeatures> features;
        std::vector<int> labels;
        
        for (const auto& sample : samples) {
            features.push_back(sample.features);
            labels.push_back(sample.label);
        }
        
        // Normalizar características (el predictor lo hace internamente)
        predictor_->normalize_features(features);
        
        // Convertir a tensores
        auto X = predictor_->features_to_tensor(features);
        auto Y = predictor_->labels_to_tensor(labels);
        
        // Entrenar modelo
        using namespace utec::neural_network;
        size_t epochs = 200;
        size_t batch_size = 32;
        double learning_rate = 0.001;
        
        std::cout << "Épocas: " << epochs << ", Batch size: " << batch_size 
                  << ", Learning rate: " << learning_rate << std::endl;
        
        predictor_->get_model()->template train<BCE, Adam>(X, Y, epochs, batch_size, learning_rate);
        
        // Marcar modelo como entrenado
        predictor_->set_trained(true);
        
        std::cout << "✓ Entrenamiento completado" << std::endl;
    }
    
    void validate_with_test_data() {
        std::cout << "\n=== VALIDACIÓN CON DATOS DE PRUEBA ===" << std::endl;
        
        size_t correct_predictions = 0;
        size_t total_predictions = 0;
        size_t true_positives = 0, false_positives = 0;
        size_t true_negatives = 0, false_negatives = 0;
        
        std::cout << std::left << std::setw(12) << "Fecha" 
                  << std::setw(8) << "Símbolo"
                  << std::setw(6) << "Real" 
                  << std::setw(10) << "Predicho"
                  << std::setw(12) << "Prob(%)"
                  << std::setw(10) << "Acierto" << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        
        for (const auto& sample : test_samples_) {
            // Hacer predicción usando el modelo entrenado
            double probability = predictor_->predict_probability(sample.features);
            int predicted = (probability >= 0.5) ? 1 : 0;
            bool correct = (predicted == sample.actual_target);
            
            if (correct) correct_predictions++;
            total_predictions++;
            
            // Actualizar matriz de confusión
            if (sample.actual_target == 1 && predicted == 1) true_positives++;
            else if (sample.actual_target == 0 && predicted == 1) false_positives++;
            else if (sample.actual_target == 0 && predicted == 0) true_negatives++;
            else if (sample.actual_target == 1 && predicted == 0) false_negatives++;
            
            std::cout << std::left << std::setw(12) << sample.date.substr(0,10)
                      << std::setw(8) << sample.symbol
                      << std::setw(6) << sample.actual_target
                      << std::setw(10) << predicted
                      << std::setw(12) << std::fixed << std::setprecision(1) << (probability * 100)
                      << std::setw(10) << (correct ? "✓" : "✗") << std::endl;
        }
        
        // Calcular métricas
        double accuracy = static_cast<double>(correct_predictions) / total_predictions;
        double precision = (true_positives + false_positives > 0) ? 
                          static_cast<double>(true_positives) / (true_positives + false_positives) : 0.0;
        double recall = (true_positives + false_negatives > 0) ? 
                       static_cast<double>(true_positives) / (true_positives + false_negatives) : 0.0;
        double f1_score = (precision + recall > 0) ? 
                         2.0 * (precision * recall) / (precision + recall) : 0.0;
        
        std::cout << "\n=== MÉTRICAS DE RENDIMIENTO ===" << std::endl;
        std::cout << "🎯 Accuracy:  " << std::fixed << std::setprecision(2) 
                  << (accuracy * 100) << "% (" << correct_predictions << "/" << total_predictions << ")" << std::endl;
        std::cout << "📊 Precision: " << (precision * 100) << "%" << std::endl;
        std::cout << "📈 Recall:    " << (recall * 100) << "%" << std::endl;
        std::cout << "🔢 F1-Score:  " << (f1_score * 100) << "%" << std::endl;
        
        std::cout << "\n📋 Matriz de Confusión:" << std::endl;
        std::cout << "                Predicho" << std::endl;
        std::cout << "                0       1" << std::endl;
        std::cout << "Real    0       " << true_negatives << "       " << false_positives << std::endl;
        std::cout << "        1       " << false_negatives << "       " << true_positives << std::endl;
        
        // Análisis detallado del rendimiento
        std::cout << "\n=== ANÁLISIS DEL MODELO ===" << std::endl;
        if (accuracy > 0.7) {
            std::cout << "✓ Excelente rendimiento del modelo" << std::endl;
        } else if (accuracy > 0.6) {
            std::cout << "⚠ Rendimiento aceptable, considerar más entrenamiento" << std::endl;
        } else if (accuracy > 0.5) {
            std::cout << "⚠ Rendimiento bajo, revisar hiperparámetros" << std::endl;
        } else {
            std::cout << "✗ Rendimiento muy bajo, modelo necesita ajustes" << std::endl;
        }
        
        // Análisis de balance
        if (std::abs(precision - recall) < 0.1) {
            std::cout << "✓ Modelo balanceado (precision ≈ recall)" << std::endl;
        } else if (precision > recall) {
            std::cout << "⚠ Modelo conservador (más precisión, menos recall)" << std::endl;
        } else {
            std::cout << "⚠ Modelo agresivo (más recall, menos precisión)" << std::endl;
        }
    }

    // Getter para acceder al modelo (necesario para el entrenamiento)
    utec::neural_network::NeuralNetwork<double>* get_model() {
        return predictor_->get_model();
    }
    
};

int main() {
    std::cout << "🤖 VALIDADOR DE PREDICTOR DE ACCIONES CON RED NEURONAL" << std::endl;
    std::cout << "=" << std::string(60, '=') << std::endl;
    
    SimpleValidator validator;
    
    // Cargar datos de entrenamiento
    std::cout << "\n📂 Cargando datos de entrenamiento..." << std::endl;
    if (!validator.load_training_data("stock_data_training.csv")) {
        std::cout << "Error: No se pudieron cargar los datos de entrenamiento" << std::endl;
        return 1;
    }
    
    // Cargar datos de prueba
    std::cout << "\n📂 Cargando datos de prueba..." << std::endl;
    if (!validator.load_test_data("stock_data_test.csv")) {
        std::cout << "Error: No se pudieron cargar los datos de prueba" << std::endl;
        return 1;
    }
    
    // Entrenar y validar
    validator.train_and_validate();
    
    std::cout << "\n🎉 Validación completada exitosamente" << std::endl;
    
    return 0;
}