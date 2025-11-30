#ifndef UTEC_NEURAL_NETWORK_ACTIVATION_H
#define UTEC_NEURAL_NETWORK_ACTIVATION_H

#include "nn_interfaces.h"
#include <cmath>
#include <algorithm>

namespace utec::neural_network {

  template<typename T>
  class ReLU final : public ILayer<T> {
  private:
    mutable Tensor<T,2> last_input_;
    
  public:
    ReLU() : last_input_(1, 1) {}
    
    Tensor<T,2> forward(const Tensor<T,2>& z) override {
      // Guardar la entrada para usar en backward
      last_input_ = z;
      
      // Crear tensor resultado con la misma forma
      Tensor<T,2> result(z.shape());
      
      // Aplicar ReLU: max(0, x)
      auto it_result = result.begin();
      for (auto it = z.begin(); it != z.end(); ++it, ++it_result) {
        *it_result = std::max(T{0}, *it);
      }
      
      return result;
    }
    
    Tensor<T,2> backward(const Tensor<T,2>& grad_output) override {
      // Crear tensor resultado con la misma forma
      Tensor<T,2> grad_input(grad_output.shape());
      
      // Derivada de ReLU: 1 si x > 0, 0 si x <= 0
      auto it_grad_input = grad_input.begin();
      auto it_grad_output = grad_output.begin();
      for (auto it_input = last_input_.begin(); 
           it_input != last_input_.end(); 
           ++it_input, ++it_grad_input, ++it_grad_output) {
        *it_grad_input = (*it_input > T{0}) ? *it_grad_output : T{0};
      }
      
      return grad_input;
    }
  };

  template<typename T>
  class Sigmoid final : public ILayer<T> {
  private:
    mutable Tensor<T,2> last_output_;
    
  public:
    Sigmoid() : last_output_(1, 1) {}
    
    Tensor<T,2> forward(const Tensor<T,2>& z) override {
      Tensor<T,2> result(z.shape());

      auto it_result = result.begin();
      for (auto it = z.begin(); it != z.end(); ++it, ++it_result) {
        T x = *it;
        if (x > T{500}) {
          *it_result = T{1};
        } else if (x < T{-500}) {
          *it_result = T{0};
        } else {
          *it_result = T{1} / (T{1} + std::exp(-x));
        }
      }
      
      // Guardar la salida para usar en backward
      last_output_ = result;
      return result;
    }
    
    Tensor<T,2> backward(const Tensor<T,2>& grad_output) override {
      // Crear tensor resultado con la misma forma
      Tensor<T,2> grad_input(grad_output.shape());
      
      // Derivada de Sigmoid: sigmoid(x) * (1 - sigmoid(x))
      auto it_grad_input = grad_input.begin();
      auto it_grad_output = grad_output.begin();
      for (auto it_output = last_output_.begin(); 
           it_output != last_output_.end(); 
           ++it_output, ++it_grad_input, ++it_grad_output) {
        T sigmoid_val = *it_output;
        *it_grad_input = *it_grad_output * sigmoid_val * (T{1} - sigmoid_val);
      }
      
      return grad_input;
    }
  };

} // namespace utec::neural_network

#endif // UTEC_NEURAL_NETWORK_ACTIVATION_H
