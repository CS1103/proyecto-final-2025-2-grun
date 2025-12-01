#include "../../../include/utec/apps/portfolio_optimizer.h"
#include "../../../include/utec/apps/data_loader.h"
#include <iostream>
#include <memory>
#include <iomanip>

int main() {
    try {
        // Cargar datos usando la nueva estructura
        utec::apps::DataLoader loader_stocks("../data/stocks");
        utec::apps::DataLoader loader_etfs("../data/etfs");
        
        // Activos seleccionados (10 total = ligero pero diversificado)
        std::vector<std::string> selected_stocks = {"AAPL", "GOOGL", "MSFT", "JNJ", "JPM"};
        std::vector<std::string> selected_etfs = {"SPY", "QQQ", "VTI", "BND", "GLD"};
        
        std::cout << "🔄 Cargando portfolio optimizado..." << std::endl;
        
        auto stock_data = loader_stocks.load_portfolio_data(selected_stocks);
        auto etf_data = loader_etfs.load_portfolio_data(selected_etfs);
        
        // Portfolio combinado
        std::vector<utec::apps::MarketData> portfolio;
        portfolio.insert(portfolio.end(), stock_data.begin(), stock_data.end());
        portfolio.insert(portfolio.end(), etf_data.begin(), etf_data.end());
        
        std::cout << "✅ Cargados " << portfolio.size() << " activos:" << std::endl;
        
        // Mostrar resumen por categoría
        std::cout << "\n📈 STOCKS (Crecimiento):" << std::endl;
        for (const auto& asset : stock_data) {
            std::cout << "  " << asset.symbol << " - Datos: " << asset.prices.size() 
                      << " días, Sharpe: " << asset.sharpe_ratio() << std::endl;
        }
        
        std::cout << "\n🏛️ ETFs (Diversificación):" << std::endl;
        for (const auto& asset : etf_data) {
            std::cout << "  " << asset.symbol << " - Datos: " << asset.prices.size() 
                      << " días, Sharpe: " << asset.sharpe_ratio() << std::endl;
        }
        
        // Mostrar información de datos cargados
        if (!portfolio.empty()) {
            size_t total_days = portfolio[0].prices.size();
            std::cout << "\n📊 Datos disponibles: " << total_days << " días de historia" << std::endl;
            std::cout << "🎯 Portfolio balanceado: " << stock_data.size() << " stocks + " << etf_data.size() << " ETFs" << std::endl;
            std::cout << "✅ Datos listos para entrenamiento de red neuronal" << std::endl;
        } else {
            std::cout << "\n⚠️ No se pudieron cargar datos del portfolio" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}