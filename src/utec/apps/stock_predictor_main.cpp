#include "../../../include/utec/apps/stock_predictor.h"
#include "../../../include/utec/apps/data_loader.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>

using namespace utec::apps;

void print_usage() {
    std::cout << "Uso del Predictor de Acciones:\n";
    std::cout << "  ./stock_predictor [modo] [opciones]\n\n";
    std::cout << "Modos:\n";
    std::cout << "  train    - Entrenar el modelo con datos históricos\n";
    std::cout << "  predict  - Hacer predicción para una acción específica\n";
    std::cout << "  evaluate - Evaluar el modelo con datos de prueba\n";
    std::cout << "  demo     - Demostración completa\n\n";
    std::cout << "Ejemplos:\n";
    std::cout << "  ./stock_predictor demo\n";
    std::cout << "  ./stock_predictor train\n";
    std::cout << "  ./stock_predictor predict AAPL\n";
}

void train_model() {
    std::cout << "=== Entrenamiento del Modelo de Predicción ===\n";
    
    try {
        // Configurar cargador de datos
        DataLoader loader("data/stocks");
        
        // Cargar símbolos de entrenamiento
        std::vector<std::string> training_symbols = {"AAPL", "MSFT", "GOOGL", "JPM"};
        
        std::cout << "Cargando datos de entrenamiento...\n";
        std::vector<MarketData> training_data;
        
        for (const auto& symbol : training_symbols) {
            try {
                std::string filename = "data/stocks/" + symbol + ".csv";
                auto data = loader.load_yahoo_csv(symbol, filename);
                loader.validate_data(data);
                training_data.push_back(data);
                std::cout << "✓ " << symbol << ": " << data.prices.size() << " puntos de datos\n";
            } catch (const std::exception& e) {
                std::cout << "✗ Error cargando " << symbol << ": " << e.what() << "\n";
            }
        }
        
        if (training_data.empty()) {
            std::cout << "Error: No se pudieron cargar datos de entrenamiento\n";
            return;
        }
        
        // Crear y configurar el predictor
        StockPredictor predictor(20, 1); // 20 días de lookback, predecir 1 día
        predictor.build_model({64, 32, 16}); // Red con 3 capas ocultas
        
        // Entrenar el modelo
        std::cout << "\nEntrenando modelo...\n";
        predictor.train(training_data, 50, 0.001, 32); // 50 épocas, lr=0.001, batch=32
        
        // Guardar modelo
        predictor.save_model("model.dat");
        std::cout << "Modelo entrenado y guardado exitosamente.\n";
        
    } catch (const std::exception& e) {
        std::cout << "Error durante el entrenamiento: " << e.what() << "\n";
    }
}

void predict_stock(const std::string& symbol) {
    std::cout << "=== Predicción para " << symbol << " ===\n";
    
    try {
        // Cargar modelo
        StockPredictor predictor;
        predictor.load_model("model.dat");
        
        // Cargar datos del símbolo
        DataLoader loader("data/stocks");
        std::string filename = "data/stocks/" + symbol + ".csv";
        auto data = loader.load_yahoo_csv(symbol, filename);
        
        // Hacer predicción
        double probability = predictor.predict_next(data);
        int prediction = predictor.predict_binary(
            predictor.extract_features(data).back(), 0.5);
        
        std::cout << "Probabilidad de subida: " << (probability * 100) << "%\n";
        std::cout << "Predicción: " << (prediction ? "SUBE ↑" : "BAJA ↓") << "\n";
        
        // Mostrar precio actual
        if (!data.prices.empty()) {
            std::cout << "Precio actual: $" << data.prices.back() << "\n";
        }
        
    } catch (const std::exception& e) {
        std::cout << "Error durante la predicción: " << e.what() << "\n";
    }
}

void evaluate_model() {
    std::cout << "=== Evaluación del Modelo ===\n";
    
    try {
        // Cargar modelo
        StockPredictor predictor;
        predictor.load_model("model.dat");
        
        // Cargar datos de prueba (diferentes a los de entrenamiento)
        DataLoader loader("data/stocks");
        std::vector<std::string> test_symbols = {"JNJ"}; // Usar JNJ para prueba
        
        std::vector<MarketData> test_data;
        for (const auto& symbol : test_symbols) {
            try {
                std::string filename = "data/stocks/" + symbol + ".csv";
                auto data = loader.load_yahoo_csv(symbol, filename);
                loader.validate_data(data);
                test_data.push_back(data);
                std::cout << "Cargado " << symbol << " para evaluación\n";
            } catch (const std::exception& e) {
                std::cout << "Error cargando " << symbol << ": " << e.what() << "\n";
            }
        }
        
        if (!test_data.empty()) {
            double accuracy = predictor.evaluate(test_data);
            std::cout << "Precisión del modelo: " << (accuracy * 100) << "%\n";
            
            if (accuracy > 0.6) {
                std::cout << "✓ El modelo muestra un rendimiento aceptable\n";
            } else {
                std::cout << "⚠ El modelo necesita más entrenamiento\n";
            }
        } else {
            std::cout << "No se pudieron cargar datos de prueba\n";
        }
        
    } catch (const std::exception& e) {
        std::cout << "Error durante la evaluación: " << e.what() << "\n";
    }
}

void demo() {
    std::cout << "=== DEMOSTRACIÓN DEL PREDICTOR DE ACCIONES ===\n";
    std::cout << "Este programa utiliza redes neuronales para predecir si una acción subirá o bajará.\n\n";
    
    // 1. Entrenar modelo
    std::cout << "1. ENTRENAMIENTO:\n";
    train_model();
    std::cout << "\n";
    
    // 2. Hacer predicciones para varias acciones
    std::cout << "2. PREDICCIONES:\n";
    std::vector<std::string> symbols = {"AAPL", "MSFT", "GOOGL", "JPM"};
    
    for (const auto& symbol : symbols) {
        predict_stock(symbol);
        std::cout << "\n";
    }
    
    // 3. Evaluar modelo
    std::cout << "3. EVALUACIÓN:\n";
    evaluate_model();
    std::cout << "\n";
    
    std::cout << "=== DEMOSTRACIÓN COMPLETADA ===\n";
    std::cout << "El modelo ha sido entrenado y evaluado con datos históricos.\n";
    std::cout << "Las predicciones se basan en indicadores técnicos como:\n";
    std::cout << "- Medias móviles (5, 10, 20 días)\n";
    std::cout << "- RSI (Índice de Fuerza Relativa)\n";
    std::cout << "- Volatilidad histórica\n";
    std::cout << "- Momentum y ratios de volumen\n";
}

void analyze_features() {
    std::cout << "=== ANÁLISIS DE CARACTERÍSTICAS TÉCNICAS ===\n";
    
    try {
        DataLoader loader("data/stocks");
        auto data = loader.load_yahoo_csv("AAPL", "data/stocks/AAPL.csv");
        
        StockPredictor predictor;
        auto features = predictor.extract_features(data);
        
        if (!features.empty()) {
            const auto& latest = features.back();
            std::cout << "Características más recientes para AAPL:\n";
            std::cout << "- Cambio 1 día: " << (latest.price_change_1d * 100) << "%\n";
            std::cout << "- Cambio 5 días: " << (latest.price_change_5d * 100) << "%\n";
            std::cout << "- RSI: " << latest.rsi << "\n";
            std::cout << "- Volatilidad: " << (latest.volatility * 100) << "%\n";
            std::cout << "- Momentum: " << (latest.momentum * 100) << "%\n";
        }
        
    } catch (const std::exception& e) {
        std::cout << "Error en análisis: " << e.what() << "\n";
    }
}

int main(int argc, char* argv[]) {
    std::cout << "Predictor de Acciones con Redes Neuronales v1.0\n";
    std::cout << "===============================================\n\n";
    
    if (argc < 2) {
        print_usage();
        return 1;
    }
    
    std::string mode = argv[1];
    
    try {
        if (mode == "demo") {
            demo();
        } else if (mode == "train") {
            train_model();
        } else if (mode == "predict") {
            if (argc < 3) {
                std::cout << "Error: Especifica el símbolo de la acción (ej: AAPL)\n";
                return 1;
            }
            predict_stock(argv[2]);
        } else if (mode == "evaluate") {
            evaluate_model();
        } else if (mode == "analyze") {
            analyze_features();
        } else {
            std::cout << "Modo desconocido: " << mode << "\n\n";
            print_usage();
            return 1;
        }
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}