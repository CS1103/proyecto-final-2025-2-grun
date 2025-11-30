#ifndef UTEC_APPS_PORTFOLIO_OPTIMIZER_H
#define UTEC_APPS_PORTFOLIO_OPTIMIZER_H

#include "../nn/neural_network.h"
#include "../algebra/tensor.h"
#include <vector>
#include <string>
#include <map>
#include <random>
#include <memory>

namespace utec::apps {

using namespace utec::neural_network;

/**
 * @brief Clase para representar datos de mercado de un activo
 */
struct MarketData {
    std::vector<double> prices;      // Precios históricos
    std::vector<double> volumes;     // Volúmenes de transacción
    std::vector<double> returns;     // Retornos calculados
    std::string symbol;              // Símbolo del activo (ej: "AAPL", "BTC")
    
    // Calcular retornos a partir de precios
    void calculate_returns();
    
    // Obtener estadísticas básicas
    double mean_return() const;
    double volatility() const;
    double sharpe_ratio(double risk_free_rate = 0.02) const;
};

/**
 * @brief Datos del portfolio en un momento específico
 */
struct PortfolioState {
    std::vector<double> asset_weights;     // Pesos actuales de cada activo (suma = 1.0)
    std::vector<double> asset_returns;     // Retornos recientes de cada activo
    std::vector<double> market_indicators; // Indicadores de mercado (volatilidad, correlaciones)
    double portfolio_value;                // Valor total del portfolio
    double cash_ratio;                     // Proporción en efectivo
    
    // Convertir estado a tensor para la red neuronal
    utec::algebra::Tensor<double, 2> to_tensor() const;
};

/**
 * @brief Ambiente de simulación de trading de portfolios
 */
class PortfolioEnvironment {
private:
    std::vector<MarketData> assets_data_;     // Datos históricos de todos los activos
    std::vector<std::string> asset_symbols_;  // Símbolos de los activos
    size_t current_time_step_;                // Tiempo actual en la simulación
    size_t lookback_period_;                  // Períodos hacia atrás para características
    double transaction_cost_;                 // Costo de transacción (%)
    double initial_capital_;                  // Capital inicial
    
    PortfolioState current_state_;
    std::mt19937 rng_;
    
public:
    explicit PortfolioEnvironment(double initial_capital = 100000.0, 
                                size_t lookback_period = 20,
                                double transaction_cost = 0.001);
    
    // Configuración del ambiente
    void add_asset(const MarketData& asset_data);
    void set_time_period(size_t start_time, size_t end_time);
    
    // Operaciones del ambiente
    PortfolioState reset();                           // Reiniciar simulación
    std::tuple<PortfolioState, double, bool> step(const std::vector<double>& new_weights);
    
    // Información del estado
    const PortfolioState& get_current_state() const { return current_state_; }
    size_t get_num_assets() const { return assets_data_.size(); }
    bool is_done() const;
    
    // Métricas de rendimiento
    double calculate_portfolio_return() const;
    double calculate_sharpe_ratio() const;
    double calculate_max_drawdown() const;
    
private:
    void update_state();
    double calculate_transaction_costs(const std::vector<double>& old_weights, 
                                     const std::vector<double>& new_weights);
};

/**
 * @brief Agente de red neuronal para optimización de portfolios
 */
class PortfolioAgent {
private:
    std::unique_ptr<NeuralNetwork<double>> network_;
    double epsilon_;                    // Para exploración
    size_t input_size_;
    size_t output_size_;
    std::mt19937 rng_;
    
    // Historial de experiencias para entrenamiento
    struct Experience {
        utec::algebra::Tensor<double, 2> state;
        std::vector<double> action;
        double reward;
        utec::algebra::Tensor<double, 2> next_state;
        bool done;
    };
    std::vector<Experience> replay_buffer_;
    size_t buffer_capacity_;
    
public:
    explicit PortfolioAgent(size_t input_size, size_t output_size, 
                          double epsilon = 0.1, size_t buffer_capacity = 10000);
    
    // Construcción de la red neuronal
    void build_network(const std::vector<size_t>& hidden_layers = {64, 32, 16});
    
    // Selección de acción
    std::vector<double> select_action(const PortfolioState& state, bool training = true);
    
    // Entrenamiento
    void store_experience(const PortfolioState& state, const std::vector<double>& action,
                         double reward, const PortfolioState& next_state, bool done);
    void train_batch(size_t batch_size = 32);
    
    // Control de exploración
    void set_epsilon(double epsilon) { epsilon_ = epsilon; }
    double get_epsilon() const { return epsilon_; }
    
    // Guardar/cargar modelo (simplificado)
    void save_model(const std::string& filename);
    void load_model(const std::string& filename);
    
private:
    std::vector<double> softmax(const std::vector<double>& logits);
    void ensure_valid_weights(std::vector<double>& weights);
};

/**
 * @brief Pipeline de entrenamiento para el optimizador de portfolios
 */
class PortfolioTrainingPipeline {
private:
    std::unique_ptr<PortfolioEnvironment> env_;
    std::unique_ptr<PortfolioAgent> agent_;
    
    // Métricas de entrenamiento
    std::vector<double> episode_returns_;
    std::vector<double> episode_sharpe_ratios_;
    std::vector<double> episode_max_drawdowns_;
    
public:
    PortfolioTrainingPipeline(std::unique_ptr<PortfolioEnvironment> env,
                            std::unique_ptr<PortfolioAgent> agent);
    
    // Entrenamiento principal
    void train(size_t num_episodes = 1000, size_t max_steps_per_episode = 252);
    
    // Evaluación
    double evaluate(size_t num_episodes = 10);
    
    // Métodos de optimización experimental
    void random_search_hyperparameters(size_t num_trials = 50);
    void hill_climbing_architecture(size_t num_iterations = 20);
    
    // Análisis de resultados
    void print_training_summary() const;
    void save_training_metrics(const std::string& filename) const;
    
    // Benchmark contra estrategias simples
    double benchmark_buy_and_hold();
    double benchmark_equal_weight();
    double benchmark_momentum_strategy();
    
private:
    void run_single_episode(size_t max_steps, bool training = true);
    std::vector<size_t> generate_random_architecture();
};

/**
 * @brief Utilidades para generar datos sintéticos y cargar datos reales
 */
class DataGenerator {
public:
    // Generar datos sintéticos de mercado
    static std::vector<MarketData> generate_synthetic_market(
        size_t num_assets = 5, 
        size_t num_days = 1000, 
        double base_return = 0.08, 
        double base_volatility = 0.15);
    
    // Cargar datos desde CSV (formato: fecha,símbolo,precio,volumen)
    static std::vector<MarketData> load_from_csv(const std::string& filename);
    
    // Generar datos con diferentes regímenes de mercado
    static std::vector<MarketData> generate_multi_regime_data(
        size_t num_assets = 5,
        size_t num_days = 1000);
    
private:
    static std::vector<double> generate_correlated_returns(
        const std::vector<std::vector<double>>& correlation_matrix,
        size_t num_samples);
};

} // namespace utec::apps

#endif // UTEC_APPS_PORTFOLIO_OPTIMIZER_H