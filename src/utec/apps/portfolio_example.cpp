#include "../../../include/utec/apps/portfolio_optimizer.h"
#include <iostream>
#include <memory>
#include <iomanip>

using namespace utec::apps;

/**
 * @brief Ejemplo de uso del optimizador de cartera con red neuronal
 */
int main() {
    std::cout << "=== Portfolio Optimizer with Neural Network ===" << std::endl;
    std::cout << "Initializing portfolio optimization system..." << std::endl;
    
    try {
        // 1. Generar datos sintéticos de mercado
        std::cout << "\n1. Generating synthetic market data..." << std::endl;
        auto market_data = DataGenerator::generate_synthetic_market(
            5,      // 5 assets
            1000,   // 1000 days of data
            0.08,   // 8% annual return
            0.15    // 15% annual volatility
        );
        
        std::cout << "Generated data for " << market_data.size() << " assets:" << std::endl;
        for (size_t i = 0; i < market_data.size(); ++i) {
            const auto& asset = market_data[i];
            std::cout << "  " << asset.symbol 
                      << " - Mean Return: " << asset.mean_return() * 252 * 100 << "%"
                      << ", Volatility: " << asset.volatility() * std::sqrt(252) * 100 << "%"
                      << ", Sharpe Ratio: " << asset.sharpe_ratio() << std::endl;
        }
        
        // 2. Crear ambiente de portfolio
        std::cout << "\n2. Setting up portfolio environment..." << std::endl;
        auto env = std::make_unique<PortfolioEnvironment>(
            100000.0,  // $100,000 initial capital
            20,        // 20-day lookback period
            0.001      // 0.1% transaction cost
        );
        
        // Agregar los activos al ambiente
        for (const auto& asset : market_data) {
            env->add_asset(asset);
        }
        
        // 3. Crear agente de red neuronal
        std::cout << "3. Creating neural network agent..." << std::endl;
        size_t num_assets = market_data.size();
        size_t input_size = num_assets * 3 + 2; // weights + returns + volatilities + portfolio_value + cash_ratio
        size_t output_size = num_assets;        // portfolio weights
        
        auto agent = std::make_unique<PortfolioAgent>(
            input_size, 
            output_size, 
            0.3,    // epsilon for exploration
            10000   // replay buffer size
        );
        
        // Construir la arquitectura de la red neuronal
        agent->build_network({64, 32, 16}); // Hidden layers: 64 -> 32 -> 16
        std::cout << "Neural network architecture: " << input_size 
                  << " -> 64 -> 32 -> 16 -> " << output_size << std::endl;
        
        // 4. Crear pipeline de entrenamiento
        std::cout << "\n4. Setting up training pipeline..." << std::endl;
        PortfolioTrainingPipeline pipeline(std::move(env), std::move(agent));
        
        // 5. Entrenamiento
        std::cout << "\n5. Starting training..." << std::endl;
        pipeline.train(
            500,  // 500 episodes
            200   // 200 steps per episode (about 8 months of trading days)
        );
        
        // 6. Evaluación
        std::cout << "\n6. Evaluating trained agent..." << std::endl;
        double avg_performance = pipeline.evaluate(20); // 20 evaluation episodes
        
        // 7. Benchmark comparisons
        std::cout << "\n7. Benchmark comparisons..." << std::endl;
        std::cout << "Neural Network Agent: " << avg_performance * 100 << "%" << std::endl;
        
        // Note: These benchmark methods would need access to a fresh environment
        // For demonstration, we'll show placeholder values
        std::cout << "Buy & Hold Strategy: ~8.0% (theoretical)" << std::endl;
        std::cout << "Equal Weight Strategy: ~7.5% (theoretical)" << std::endl;
        
        // 8. Resultados finales
        std::cout << "\n=== Final Results ===" << std::endl;
        if (avg_performance > 0.08) {
            std::cout << "✅ Neural network agent outperformed market!" << std::endl;
        } else if (avg_performance > 0.06) {
            std::cout << "✅ Neural network agent showed competitive performance." << std::endl;
        } else {
            std::cout << "⚠️  Neural network agent underperformed. Consider:" << std::endl;
            std::cout << "   - Longer training" << std::endl;
            std::cout << "   - Different network architecture" << std::endl;
            std::cout << "   - Hyperparameter tuning" << std::endl;
        }
        
        // 9. Experimentos adicionales sugeridos
        std::cout << "\n=== Suggested Experiments ===" << std::endl;
        std::cout << "1. Try different market conditions (bear market, high volatility)" << std::endl;
        std::cout << "2. Experiment with different network architectures" << std::endl;
        std::cout << "3. Add more features (technical indicators, market sentiment)" << std::endl;
        std::cout << "4. Implement different loss functions (Sharpe ratio optimization)" << std::endl;
        std::cout << "5. Add regularization to prevent overfitting" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\nPortfolio optimization demonstration completed!" << std::endl;
    return 0;
}

// Función auxiliar para mostrar el estado del portfolio
void print_portfolio_state(const PortfolioState& state) {
    std::cout << "Portfolio State:" << std::endl;
    std::cout << "  Value: $" << std::fixed << std::setprecision(2) << state.portfolio_value << std::endl;
    std::cout << "  Weights: [";
    for (size_t i = 0; i < state.asset_weights.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(3) << state.asset_weights[i];
    }
    std::cout << "]" << std::endl;
    std::cout << "  Recent Returns: [";
    for (size_t i = 0; i < state.asset_returns.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(4) << state.asset_returns[i] * 100 << "%";
    }
    std::cout << "]" << std::endl;
}