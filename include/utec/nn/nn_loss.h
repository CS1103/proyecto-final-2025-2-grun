#ifndef UTEC_NEURAL_NETWORK_LOSS_H
#define UTEC_NEURAL_NETWORK_LOSS_H

#include "nn_interfaces.h"
#include <cmath>
#include <algorithm>

namespace utec::neural_network {

  template<typename T>
  class MSELoss final: public ILoss<T, 2> {
  private:
    Tensor<T,2> y_prediction_;
    Tensor<T,2> y_true_;
    
  public:
    MSELoss(const Tensor<T,2>& y_prediction, const Tensor<T,2>& y_true) 
        : y_prediction_(y_prediction), y_true_(y_true) {}
    
    T loss() const override {
      T total_loss = T{0};
      size_t count = 0;
      
      auto it_pred = y_prediction_.begin();
      auto it_true = y_true_.begin();
      
      for (; it_pred != y_prediction_.end(); ++it_pred, ++it_true, ++count) {
        T diff = (*it_pred - *it_true);
        total_loss += diff * diff;
      }
      
      return total_loss / static_cast<T>(count);
    }
    
    Tensor<T,2> loss_gradient() const override {
      Tensor<T,2> gradient(y_prediction_.shape());
      
      T n = static_cast<T>(y_prediction_.size());
      auto it_grad = gradient.begin();
      auto it_pred = y_prediction_.begin();
      auto it_true = y_true_.begin();
      
      for (; it_grad != gradient.end(); ++it_grad, ++it_pred, ++it_true) {
        *it_grad = (T{2} * (*it_pred - *it_true)) / n;
      }
      
      return gradient;
    }
  };

  template<typename T>
  class BCELoss final: public ILoss<T, 2> {
  private:
    Tensor<T,2> y_prediction_;
    Tensor<T,2> y_true_;
    
  public:
    BCELoss(const Tensor<T,2>& y_prediction, const Tensor<T,2>& y_true) 
        : y_prediction_(y_prediction), y_true_(y_true) {}
    
    T loss() const override {
      T total_loss = T{0};
      size_t count = 0;
      
      auto it_pred = y_prediction_.begin();
      auto it_true = y_true_.begin();
      
      for (; it_pred != y_prediction_.end(); ++it_pred, ++it_true, ++count) {
        T pred = *it_pred;
        T true_val = *it_true;
        
        pred = std::max(T{1e-15}, std::min(T{1.0 - 1e-15}, pred));
        
        total_loss += -(true_val * std::log(pred) + (T{1} - true_val) * std::log(T{1} - pred));
      }
      
      return total_loss / static_cast<T>(count);
    }
    
    Tensor<T,2> loss_gradient() const override {
      Tensor<T,2> gradient(y_prediction_.shape());
      
      T n = static_cast<T>(y_prediction_.size());
      auto it_grad = gradient.begin();
      auto it_pred = y_prediction_.begin();
      auto it_true = y_true_.begin();
      
      for (; it_grad != gradient.end(); ++it_grad, ++it_pred, ++it_true) {
        T pred = *it_pred;
        T true_val = *it_true;
        
        pred = std::max(T{1e-15}, std::min(T{1.0 - 1e-15}, pred));
        
        *it_grad = ((pred - true_val) / (pred * (T{1} - pred))) / n;
      }
      
      return gradient;
    }
  };

  // Alias para facilitar el uso
  template<typename T>
  using MSE = MSELoss<T>;
  
  template<typename T>
  using BCE = BCELoss<T>;

} // namespace utec::neural_network

#endif // UTEC_NEURAL_NETWORK_LOSS_H
