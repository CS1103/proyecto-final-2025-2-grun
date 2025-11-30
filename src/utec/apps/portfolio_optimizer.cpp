#include "../../../include/utec/apps/portfolio_optimizer.h"
#include "../../../include/utec/apps/portfolio_utils.h"
#include "../../../include/utec/nn/nn_dense.h"
#include "../../../include/utec/nn/nn_activation.h"
#include "../../../include/utec/nn/nn_loss.h"
#include "../../../include/utec/nn/nn_optimizer.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>

namespace utec::apps {

// ================= MarketData Implementation =================

void MarketData::calculate_returns() {
    returns.clear();
    if (prices.size() < 2) return;
    
    returns.reserve(prices.size() - 1);
    for (size_t i = 1; i < prices.size(); ++i) {
        double return_val = (prices[i] - prices[i-1]) / prices[i-1];
        returns.push_back(return_val);
    }
}

double MarketData::mean_return() const {
    if (returns.empty()) return 0.0;
    return std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
}

double MarketData::volatility() const {
    if (returns.size() < 2) return 0.0;
    
    double mean = mean_return();
    double variance = 0.0;
    for (double ret : returns) {
        variance += (ret - mean) * (ret - mean);
    }
    variance /= (returns.size() - 1);
    return std::sqrt(variance);
}

double MarketData::sharpe_ratio(double risk_free_rate) const {
    double vol = volatility();
    if (vol == 0.0) return 0.0;
    
    double excess_return = mean_return() * 252 - risk_free_rate; // Anualizado
    double annualized_vol = vol * std::sqrt(252);
    return excess_return / annualized_vol;
}

// ================= PortfolioState Implementation =================

utec::algebra::Tensor<double, 2> PortfolioState::to_tensor() const {
    size_t total_features = asset_weights.size() + asset_returns.size() + 
                           market_indicators.size() + 2; // +2 for portfolio_value, cash_ratio
    
    utec::algebra::Tensor<double, 2> tensor(1, total_features);
    size_t idx = 0;
    
    // Asset weights
    for (double weight : asset_weights) {
        tensor(0, idx++) = weight;
    }
    
    // Asset returns
    for (double ret : asset_returns) {
        tensor(0, idx++) = ret;
    }
    
    // Market indicators
    for (double indicator : market_indicators) {
        tensor(0, idx++) = indicator;
    }
    
    // Portfolio value (normalized)
    tensor(0, idx++) = std::log(portfolio_value / 100000.0); // Log-normalize around initial capital
    
    // Cash ratio
    tensor(0, idx++) = cash_ratio;
    
    return tensor;
}

// ================= PortfolioEnvironment Implementation =================

PortfolioEnvironment::PortfolioEnvironment(double initial_capital, size_t lookback_period, double transaction_cost)
    : current_time_step_(0), lookback_period_(lookback_period), transaction_cost_(transaction_cost), 
      initial_capital_(initial_capital), rng_(std::random_device{}()) {
}

void PortfolioEnvironment::add_asset(const MarketData& asset_data) {
    assets_data_.push_back(asset_data);
    asset_symbols_.push_back(asset_data.symbol);
}

PortfolioState PortfolioEnvironment::reset() {
    current_time_step_ = lookback_period_; // Start after lookback period
    
    // Initialize equal weights
    size_t num_assets = assets_data_.size();
    current_state_.asset_weights.assign(num_assets, 1.0 / num_assets);
    current_state_.portfolio_value = initial_capital_;
    current_state_.cash_ratio = 0.0;
    
    update_state();
    return current_state_;
}

std::tuple<PortfolioState, double, bool> PortfolioEnvironment::step(const std::vector<double>& new_weights) {
    if (new_weights.size() != assets_data_.size()) {
        throw std::invalid_argument("New weights size must match number of assets");
    }
    
    // Normalize weights to sum to 1
    std::vector<double> normalized_weights = new_weights;
    double sum = std::accumulate(normalized_weights.begin(), normalized_weights.end(), 0.0);
    if (sum > 0) {
        for (double& w : normalized_weights) {
            w /= sum;
        }
    }
    
    // Calculate transaction costs
    double transaction_cost = calculate_transaction_costs(current_state_.asset_weights, normalized_weights);
    
    // Store old portfolio value
    double old_portfolio_value = current_state_.portfolio_value;
    
    // Update weights
    current_state_.asset_weights = normalized_weights;
    
    // Apply transaction costs
    current_state_.portfolio_value *= (1.0 - transaction_cost);
    
    // Move to next time step
    current_time_step_++;
    
    // Update state with new market data
    update_state();
    
    // Calculate reward (portfolio return)
    double reward = (current_state_.portfolio_value - old_portfolio_value) / old_portfolio_value;
    
    bool done = is_done();
    
    return std::make_tuple(current_state_, reward, done);
}

void PortfolioEnvironment::update_state() {
    size_t num_assets = assets_data_.size();
    
    // Calculate recent returns for each asset
    current_state_.asset_returns.clear();
    current_state_.market_indicators.clear();
    
    for (size_t i = 0; i < num_assets; ++i) {
        if (current_time_step_ < assets_data_[i].returns.size()) {
            // Recent return
            current_state_.asset_returns.push_back(assets_data_[i].returns[current_time_step_]);
            
            // Update portfolio value based on asset performance
            if (current_time_step_ > lookback_period_) {
                double asset_return = assets_data_[i].returns[current_time_step_];
                current_state_.portfolio_value *= (1.0 + asset_return * current_state_.asset_weights[i]);
            }
        } else {
            current_state_.asset_returns.push_back(0.0);
        }
    }
    
    // Calculate market indicators (volatility, correlations)
    if (current_time_step_ >= lookback_period_) {
        // Calculate rolling volatility for each asset
        for (size_t i = 0; i < num_assets; ++i) {
            std::vector<double> recent_returns;
            size_t start_idx = current_time_step_ - lookback_period_;
            for (size_t j = start_idx; j < current_time_step_ && j < assets_data_[i].returns.size(); ++j) {
                recent_returns.push_back(assets_data_[i].returns[j]);
            }
            
            // Calculate volatility
            if (recent_returns.size() > 1) {
                double mean = std::accumulate(recent_returns.begin(), recent_returns.end(), 0.0) / recent_returns.size();
                double variance = 0.0;
                for (double ret : recent_returns) {
                    variance += (ret - mean) * (ret - mean);
                }
                variance /= (recent_returns.size() - 1);
                current_state_.market_indicators.push_back(std::sqrt(variance));
            } else {
                current_state_.market_indicators.push_back(0.0);
            }
        }
    }
    
    // Update cash ratio (simplified: assume no cash holding for now)
    current_state_.cash_ratio = 0.0;
}

bool PortfolioEnvironment::is_done() const {
    // Check if we've reached the end of available data
    for (const auto& asset : assets_data_) {
        if (current_time_step_ >= asset.returns.size()) {
            return true;
        }
    }
    return false;
}

double PortfolioEnvironment::calculate_transaction_costs(const std::vector<double>& old_weights, 
                                                       const std::vector<double>& new_weights) {
    double total_change = 0.0;
    for (size_t i = 0; i < old_weights.size(); ++i) {
        total_change += std::abs(new_weights[i] - old_weights[i]);
    }
    return total_change * transaction_cost_;
}

double PortfolioEnvironment::calculate_portfolio_return() const {
    return (current_state_.portfolio_value - initial_capital_) / initial_capital_;
}

// ================= PortfolioAgent Implementation =================

PortfolioAgent::PortfolioAgent(size_t input_size, size_t output_size, double epsilon, size_t buffer_capacity)
    : epsilon_(epsilon), input_size_(input_size), output_size_(output_size), 
      buffer_capacity_(buffer_capacity), rng_(std::random_device{}()) {
    network_ = std::make_unique<NeuralNetwork<double>>();
    replay_buffer_.reserve(buffer_capacity_);
}

void PortfolioAgent::build_network(const std::vector<size_t>& hidden_layers) {
    NetworkBuilder<double> builder(input_size_, 42);
    
    // Add hidden layers with ReLU activations
    for (size_t layer_size : hidden_layers) {
        builder.add_dense(layer_size, true).add_relu();
    }
    
    // Output layer with softmax
    builder.add_dense(output_size_, false).add_softmax();
    
    network_ = builder.build();
}

std::vector<double> PortfolioAgent::select_action(const PortfolioState& state, bool training) {
    utec::algebra::Tensor<double, 2> state_tensor = state.to_tensor();
    utec::algebra::Tensor<double, 2> action_probs = network_->predict(state_tensor);
    
    std::vector<double> weights;
    weights.reserve(output_size_);
    
    if (training && std::uniform_real_distribution<double>(0.0, 1.0)(rng_) < epsilon_) {
        // Random exploration
        std::uniform_real_distribution<double> dist(0.1, 1.0);
        for (size_t i = 0; i < output_size_; ++i) {
            weights.push_back(dist(rng_));
        }
    } else {
        // Use network prediction
        for (size_t i = 0; i < output_size_; ++i) {
            weights.push_back(action_probs(0, i));
        }
    }
    
    ensure_valid_weights(weights);
    return weights;
}

void PortfolioAgent::store_experience(const PortfolioState& state, const std::vector<double>& action,
                                    double reward, const PortfolioState& next_state, bool done) {
    Experience exp;
    exp.state = state.to_tensor();
    exp.action = action;
    exp.reward = reward;
    exp.next_state = next_state.to_tensor();
    exp.done = done;
    
    if (replay_buffer_.size() >= buffer_capacity_) {
        replay_buffer_.erase(replay_buffer_.begin());
    }
    replay_buffer_.push_back(exp);
}

void PortfolioAgent::train_batch(size_t batch_size) {
    if (replay_buffer_.size() < batch_size) return;
    
    // Simple training approach: use recent experiences
    size_t start_idx = replay_buffer_.size() - batch_size;
    
    // Create batch tensors
    utec::algebra::Tensor<double, 2> batch_states(batch_size, input_size_);
    utec::algebra::Tensor<double, 2> batch_targets(batch_size, output_size_);
    
    for (size_t i = 0; i < batch_size; ++i) {
        const Experience& exp = replay_buffer_[start_idx + i];
        
        // Copy state
        for (size_t j = 0; j < input_size_; ++j) {
            batch_states(i, j) = exp.state(0, j);
        }
        
        // Create target based on reward (simplified policy gradient approach)
        double reward_signal = exp.reward > 0 ? 1.1 : 0.9; // Amplify or reduce action probabilities
        for (size_t j = 0; j < output_size_; ++j) {
            batch_targets(i, j) = exp.action[j] * reward_signal;
        }
    }
    
    // Normalize targets
    for (size_t i = 0; i < batch_size; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < output_size_; ++j) {
            sum += batch_targets(i, j);
        }
        if (sum > 0) {
            for (size_t j = 0; j < output_size_; ++j) {
                batch_targets(i, j) /= sum;
            }
        }
    }
    
    // Train network
    network_->template train<MSE, SGD>(batch_states, batch_targets, 1, batch_size, 0.001);
}

std::vector<double> PortfolioAgent::softmax(const std::vector<double>& logits) {
    std::vector<double> result;
    result.reserve(logits.size());
    
    double max_logit = *std::max_element(logits.begin(), logits.end());
    double sum = 0.0;
    
    for (double logit : logits) {
        double exp_val = std::exp(logit - max_logit);
        result.push_back(exp_val);
        sum += exp_val;
    }
    
    for (double& val : result) {
        val /= sum;
    }
    
    return result;
}

void PortfolioAgent::ensure_valid_weights(std::vector<double>& weights) {
    // Ensure all weights are positive
    for (double& w : weights) {
        w = std::max(w, 0.01); // Minimum weight
    }
    
    // Normalize to sum to 1
    double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
    for (double& w : weights) {
        w /= sum;
    }
}

void PortfolioAgent::save_model(const std::string& filename) {
    std::cout << "Model saving not implemented yet. Filename: " << filename << std::endl;
}

void PortfolioAgent::load_model(const std::string& filename) {
    std::cout << "Model loading not implemented yet. Filename: " << filename << std::endl;
}

// ================= PortfolioTrainingPipeline Implementation =================

PortfolioTrainingPipeline::PortfolioTrainingPipeline(std::unique_ptr<PortfolioEnvironment> env,
                                                    std::unique_ptr<PortfolioAgent> agent)
    : env_(std::move(env)), agent_(std::move(agent)) {
}

void PortfolioTrainingPipeline::train(size_t num_episodes, size_t max_steps_per_episode) {
    std::cout << "Starting portfolio optimization training..." << std::endl;
    std::cout << "Episodes: " << num_episodes << ", Max steps per episode: " << max_steps_per_episode << std::endl;
    
    for (size_t episode = 0; episode < num_episodes; ++episode) {
        run_single_episode(max_steps_per_episode, true);
        
        // Decay exploration
        if (episode % 100 == 0) {
            double new_epsilon = agent_->get_epsilon() * 0.95;
            agent_->set_epsilon(std::max(new_epsilon, 0.01));
        }
        
        // Print progress
        if (episode % 100 == 0) {
            std::cout << "Episode " << episode << "/" << num_episodes 
                      << ", Epsilon: " << agent_->get_epsilon() << std::endl;
            if (!episode_returns_.empty()) {
                std::cout << "  Recent return: " << episode_returns_.back() * 100 << "%" << std::endl;
            }
        }
    }
    
    std::cout << "Training completed!" << std::endl;
    print_training_summary();
}

void PortfolioTrainingPipeline::run_single_episode(size_t max_steps, bool training) {
    PortfolioState state = env_->reset();
    double episode_return = 0.0;
    
    for (size_t step = 0; step < max_steps && !env_->is_done(); ++step) {
        // Select action
        std::vector<double> action = agent_->select_action(state, training);
        
        // Execute action
        auto [next_state, reward, done] = env_->step(action);
        
        // Store experience and train
        if (training) {
            agent_->store_experience(state, action, reward, next_state, done);
            if (step % 10 == 0) { // Train every 10 steps
                agent_->train_batch(32);
            }
        }
        
        episode_return += reward;
        state = next_state;
        
        if (done) break;
    }
    
    // Store episode metrics
    episode_returns_.push_back(env_->calculate_portfolio_return());
    episode_sharpe_ratios_.push_back(0.0); // Placeholder
    episode_max_drawdowns_.push_back(0.0); // Placeholder
}

double PortfolioTrainingPipeline::evaluate(size_t num_episodes) {
    std::cout << "Evaluating agent performance..." << std::endl;
    
    double old_epsilon = agent_->get_epsilon();
    agent_->set_epsilon(0.0); // No exploration during evaluation
    
    std::vector<double> eval_returns;
    for (size_t i = 0; i < num_episodes; ++i) {
        run_single_episode(252, false); // 252 trading days
        eval_returns.push_back(env_->calculate_portfolio_return());
    }
    
    agent_->set_epsilon(old_epsilon); // Restore exploration
    
    double avg_return = std::accumulate(eval_returns.begin(), eval_returns.end(), 0.0) / eval_returns.size();
    std::cout << "Average evaluation return: " << avg_return * 100 << "%" << std::endl;
    
    return avg_return;
}

void PortfolioTrainingPipeline::print_training_summary() const {
    if (episode_returns_.empty()) {
        std::cout << "No training data available." << std::endl;
        return;
    }
    
    double avg_return = std::accumulate(episode_returns_.begin(), episode_returns_.end(), 0.0) / episode_returns_.size();
    double best_return = *std::max_element(episode_returns_.begin(), episode_returns_.end());
    double worst_return = *std::min_element(episode_returns_.begin(), episode_returns_.end());
    
    std::cout << "\n=== Training Summary ===" << std::endl;
    std::cout << "Total episodes: " << episode_returns_.size() << std::endl;
    std::cout << "Average return: " << avg_return * 100 << "%" << std::endl;
    std::cout << "Best return: " << best_return * 100 << "%" << std::endl;
    std::cout << "Worst return: " << worst_return * 100 << "%" << std::endl;
}

// ================= DataGenerator Implementation =================

std::vector<MarketData> DataGenerator::generate_synthetic_market(size_t num_assets, size_t num_days, 
                                                               double base_return, double base_volatility) {
    std::vector<MarketData> market_data;
    std::mt19937 rng(42); // Fixed seed for reproducibility
    std::normal_distribution<double> norm_dist(0.0, 1.0);
    
    for (size_t asset = 0; asset < num_assets; ++asset) {
        MarketData data;
        data.symbol = "ASSET_" + std::to_string(asset);
        
        double price = 100.0; // Starting price
        double daily_return = base_return / 252.0; // Convert annual to daily
        double daily_vol = base_volatility / std::sqrt(252.0);
        
        data.prices.reserve(num_days);
        data.volumes.reserve(num_days);
        
        for (size_t day = 0; day < num_days; ++day) {
            // Generate correlated random walk
            double random_shock = norm_dist(rng) * daily_vol;
            double return_today = daily_return + random_shock;
            
            price *= (1.0 + return_today);
            data.prices.push_back(price);
            
            // Generate volume (simplified)
            double volume = 1000000 * (1.0 + std::abs(random_shock) * 2.0);
            data.volumes.push_back(volume);
        }
        
        data.calculate_returns();
        market_data.push_back(data);
    }
    
    return market_data;
}

std::vector<MarketData> DataGenerator::load_from_csv(const std::string& filename) {
    std::vector<MarketData> market_data;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cout << "Warning: Could not open file " << filename << ". Generating synthetic data instead." << std::endl;
        return generate_synthetic_market();
    }
    
    // Implementation for CSV parsing would go here
    // For now, return synthetic data
    file.close();
    return generate_synthetic_market();
}

std::vector<MarketData> DataGenerator::generate_multi_regime_data(size_t num_assets, size_t num_days) {
    // Generate data with different market regimes (bull, bear, sideways)
    return generate_synthetic_market(num_assets, num_days, 0.10, 0.20); // Higher volatility
}

} // namespace utec::apps