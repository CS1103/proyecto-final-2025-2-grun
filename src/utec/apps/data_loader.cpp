#include "../../../include/utec/apps/data_loader.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>

namespace utec::apps {

MarketData DataLoader::load_yahoo_csv(const std::string& symbol, const std::string& filename) {
    MarketData data;
    data.symbol = symbol;
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("No se pudo abrir el archivo: " + filename);
    }
    
    std::string line;
    bool is_header = true;
    
    while (std::getline(file, line)) {
        if (is_header) {
            is_header = false;
            continue; // Saltar encabezado: Date,Open,High,Low,Close,Adj Close,Volume
        }
        
        auto fields = split_csv_line(line);
        if (fields.size() >= 7) {
            try {
                // Formato: Date,Open,High,Low,Close,Adj Close,Volume
                //          [0]  [1]  [2] [3]  [4]   [5]        [6]
                double close_price = parse_double(fields[4]);      // Close (índice 4)
                double adj_close = parse_double(fields[5]);        // Adj Close (índice 5) 
                double volume = parse_double(fields[6]);           // Volume (índice 6)
                
                if (is_valid_price(adj_close) && volume > 0) {
                    data.prices.push_back(adj_close);  // Usar Adj Close para precios
                    data.volumes.push_back(volume);
                }
            } catch (const std::exception& e) {
                std::cerr << "Error procesando línea: " << line << " - " << e.what() << std::endl;
            }
        } else {
            std::cerr << "Línea con formato incorrecto (faltan campos): " << line << std::endl;
        }
    }
    
    // Calcular retornos
    data.calculate_returns();
    
    std::cout << "Cargados " << data.prices.size() << " datos para " << symbol << std::endl;
    return data;
}

std::vector<MarketData> DataLoader::load_portfolio_data(const std::vector<std::string>& symbols) {
    std::vector<MarketData> portfolio_data;
    
    for (const auto& symbol : symbols) {
        try {
            std::string filename = data_directory_ + "/" + symbol + ".csv";
            auto data = load_yahoo_csv(symbol, filename);
            portfolio_data.push_back(std::move(data));
        } catch (const std::exception& e) {
            std::cerr << "Error cargando datos para " << symbol << ": " << e.what() << std::endl;
        }
    }
    
    return portfolio_data;
}

MarketData DataLoader::load_alpha_vantage_csv(const std::string& symbol, const std::string& filename) {
    MarketData data;
    data.symbol = symbol;
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("No se pudo abrir el archivo: " + filename);
    }
    
    std::string line;
    bool is_header = true;
    
    while (std::getline(file, line)) {
        if (is_header) {
            is_header = false;
            continue;
        }
        
        auto fields = split_csv_line(line);
        if (fields.size() >= 6) {
            try {
                // Format: timestamp,open,high,low,close,volume
                double close_price = parse_double(fields[4]);
                double volume = parse_double(fields[5]);
                
                if (is_valid_price(close_price) && volume > 0) {
                    data.prices.push_back(close_price);
                    data.volumes.push_back(volume);
                }
            } catch (const std::exception& e) {
                std::cerr << "Error procesando línea Alpha Vantage: " << line << std::endl;
            }
        }
    }
    
    // Los datos de Alpha Vantage vienen en orden descendente, invertir
    std::reverse(data.prices.begin(), data.prices.end());
    std::reverse(data.volumes.begin(), data.volumes.end());
    
    data.calculate_returns();
    
    return data;
}

void DataLoader::validate_data(MarketData& data) {
    // Verificación básica: solo comprobar que tenemos datos mínimos
    if (data.prices.size() < 10) {
        throw std::runtime_error("Datos insuficientes para " + data.symbol + 
                               ": solo " + std::to_string(data.prices.size()) + " puntos");
    }
}

std::vector<std::string> DataLoader::split_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;
    
    while (std::getline(ss, field, ',')) {
        // Limpiar espacios en blanco
        field.erase(0, field.find_first_not_of(" \t"));
        field.erase(field.find_last_not_of(" \t") + 1);
        fields.push_back(field);
    }
    
    return fields;
}

double DataLoader::parse_double(const std::string& str) {
    if (str.empty() || str == "null" || str == "N/A") {
        throw std::invalid_argument("Valor inválido: " + str);
    }
    return std::stod(str);
}

bool DataLoader::is_valid_price(double price) {
    return price > 0 && std::isfinite(price); // Solo verificar que sea positivo y válido
}

} // namespace utec::apps