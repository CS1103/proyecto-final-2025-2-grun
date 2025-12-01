#ifndef UTEC_APPS_DATA_LOADER_H
#define UTEC_APPS_DATA_LOADER_H

#include "portfolio_optimizer.h"
#include <fstream>
#include <sstream>
#include <map>

namespace utec::apps {

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
    
    // Validar y limpiar datos
    void validate_data(MarketData& data);
    
private:
    std::vector<std::string> split_csv_line(const std::string& line);
    double parse_double(const std::string& str);
    bool is_valid_price(double price);
};

} // namespace utec::apps

#endif // UTEC_APPS_DATA_LOADER_H