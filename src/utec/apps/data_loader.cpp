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

std::vector<MarketData> DataLoader::load_features_csv(const std::string& filename) {
    std::map<std::string, MarketData> data_by_symbol;
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("No se pudo abrir el archivo: " + filename);
    }
    
    std::string line;
    bool is_header = true;
    
    while (std::getline(file, line)) {
        if (is_header) {
            is_header = false;
            continue; // Saltar encabezado
        }
        
        auto fields = split_csv_line(line);
        if (fields.size() >= 14) {
            try {
                // Formato: Date,Symbol,Close,price_change_1d,price_change_3d,price_change_5d,
                //          sma_5,sma_10,sma_20,rsi,volume_ratio,volatility,momentum,target
                std::string date = fields[0];
                std::string symbol = fields[1];
                double close = parse_double(fields[2]);
                double price_change_1d = parse_double(fields[3]);
                double price_change_3d = parse_double(fields[4]);
                double price_change_5d = parse_double(fields[5]);
                double sma_5 = parse_double(fields[6]);
                double sma_10 = parse_double(fields[7]);
                double sma_20 = parse_double(fields[8]);
                double rsi = parse_double(fields[9]);
                double volume_ratio = parse_double(fields[10]);
                double volatility = parse_double(fields[11]);
                double momentum = parse_double(fields[12]);
                int target = std::stoi(fields[13]);
                
                // Crear entrada si no existe
                if (data_by_symbol.find(symbol) == data_by_symbol.end()) {
                    data_by_symbol[symbol].symbol = symbol;
                }
                
                auto& data = data_by_symbol[symbol];
                data.dates.push_back(date);
                data.prices.push_back(close);
                data.price_change_1d.push_back(price_change_1d);
                data.price_change_3d.push_back(price_change_3d);
                data.price_change_5d.push_back(price_change_5d);
                data.sma_5.push_back(sma_5);
                data.sma_10.push_back(sma_10);
                data.sma_20.push_back(sma_20);
                data.rsi.push_back(rsi);
                data.volume_ratio.push_back(volume_ratio);
                data.volatility.push_back(volatility);
                data.momentum.push_back(momentum);
                data.target.push_back(target);
                
            } catch (const std::exception& e) {
                std::cerr << "Error procesando línea: " << line << " - " << e.what() << std::endl;
            }
        }
    }
    
    // Convertir mapa a vector
    std::vector<MarketData> result;
    for (auto& pair : data_by_symbol) {
        std::cout << "Cargados " << pair.second.prices.size() 
                  << " datos para " << pair.first << std::endl;
        result.push_back(std::move(pair.second));
    }
    
    return result;
}

} // namespace utec::apps