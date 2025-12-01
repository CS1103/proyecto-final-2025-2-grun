#ifndef UTEC_APPS_DATA_LOADER_H
#define UTEC_APPS_DATA_LOADER_H

#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <string>

namespace utec::apps {

// Estructura para almacenar datos de mercado
struct MarketData {
    std::string symbol;
    std::vector<double> prices;
    std::vector<double> volumes;
    std::vector<double> returns;
    
    // Características técnicas (si están disponibles)
    std::vector<double> price_change_1d;
    std::vector<double> price_change_3d;
    std::vector<double> price_change_5d;
    std::vector<double> sma_5;
    std::vector<double> sma_10;
    std::vector<double> sma_20;
    std::vector<double> rsi;
    std::vector<double> volume_ratio;
    std::vector<double> volatility;
    std::vector<double> momentum;
    std::vector<int> target;
    std::vector<std::string> dates;
    
    void calculate_returns() {
        returns.clear();
        for (size_t i = 1; i < prices.size(); ++i) {
            if (prices[i-1] != 0) {
                returns.push_back((prices[i] - prices[i-1]) / prices[i-1]);
            } else {
                returns.push_back(0.0);
            }
        }
    }
};

class DataLoader {
private:
    std::string data_directory_;
    
public:
    DataLoader(const std::string& data_dir) : data_directory_(data_dir) {}
    
    // Cargar datos de un archivo CSV de Yahoo Finance
    MarketData load_yahoo_csv(const std::string& symbol, const std::string& filename);
    
    // Cargar múltiples activos desde un directorio
    std::vector<MarketData> load_portfolio_data(const std::vector<std::string>& symbols);
    
    // Cargar datos desde formato Alpha Vantage
    MarketData load_alpha_vantage_csv(const std::string& symbol, const std::string& filename);
    
    // Cargar datos desde CSV con características técnicas ya calculadas
    // Formato: Date,Symbol,Close,price_change_1d,...,target
    std::vector<MarketData> load_features_csv(const std::string& filename);
    
    // Validar y limpiar datos
    void validate_data(MarketData& data);
    
private:
    std::vector<std::string> split_csv_line(const std::string& line);
    double parse_double(const std::string& str);
    bool is_valid_price(double price);
};

} // namespace utec::apps

#endif // UTEC_APPS_DATA_LOADER_H