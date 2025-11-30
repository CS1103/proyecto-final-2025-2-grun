#ifndef UTEC_NEURAL_NETWORK_OPTIMIZER_H
#define UTEC_NEURAL_NETWORK_OPTIMIZER_H

#include "nn_interfaces.h"
#include <cmath>
#include <unordered_map>

namespace utec::neural_network {

  template<typename T>
  class SGD final : public IOptimizer<T> {
  private:
    T learning_rate_;
    
  public:
    explicit SGD(T learning_rate = T{0.01}) : learning_rate_(learning_rate) {}
    
    void update(Tensor<T, 2>& params, const Tensor<T, 2>& grads) override {
      auto it_params = params.begin();
      auto it_grads = grads.begin();
      
      for (; it_params != params.end(); ++it_params, ++it_grads) {
        *it_params -= learning_rate_ * (*it_grads);
      }
    }
  };

  template<typename T>
  class Adam final : public IOptimizer<T> {
  private:
    T learning_rate_;
    T beta1_;
    T beta2_;
    T epsilon_;
    T t_;
    
    std::unordered_map<void*, Tensor<T,2>> m_;
    std::unordered_map<void*, Tensor<T,2>> v_;
    
  public:
    explicit Adam(T learning_rate = T{0.001}, T beta1 = T{0.9}, T beta2 = T{0.999}, T epsilon = T{1e-8}) 
        : learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), t_(T{0}) {}
    
    void update(Tensor<T, 2>& params, const Tensor<T, 2>& grads) override {
      void* param_id = static_cast<void*>(&params);
      
      t_ += T{1};
      
      if (m_.find(param_id) == m_.end()) {
        m_[param_id] = Tensor<T,2>(params.shape());
        v_[param_id] = Tensor<T,2>(params.shape());
        // Inicializar con ceros
        for (auto& val : m_[param_id]) val = T{0};
        for (auto& val : v_[param_id]) val = T{0};
      }
      
      auto& m = m_[param_id];
      auto& v = v_[param_id];
      
      auto it_params = params.begin();
      auto it_grads = grads.begin();
      auto it_m = m.begin();
      auto it_v = v.begin();
      
      for (; it_params != params.end(); ++it_params, ++it_grads, ++it_m, ++it_v) {
        *it_m = beta1_ * (*it_m) + (T{1} - beta1_) * (*it_grads);
        
        *it_v = beta2_ * (*it_v) + (T{1} - beta2_) * (*it_grads) * (*it_grads);
        
        T m_hat = (*it_m) / (T{1} - std::pow(beta1_, t_));
        
        T v_hat = (*it_v) / (T{1} - std::pow(beta2_, t_));
        
        *it_params -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
      }
    }
    
    void step() override {
      
    }
  };

} // namespace utec::neural_network

#endif // UTEC_NEURAL_NETWORK_OPTIMIZER_H

