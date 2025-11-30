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

  template<typename T>
  class Softmax final : public ILayer<T> {
  private:
    mutable Tensor<T,2> last_output_;
    
  public:
    Softmax() : last_output_(1, 1) {}
    
    Tensor<T,2> forward(const Tensor<T,2>& z) override {
      const auto& shape = z.shape();
      Tensor<T,2> result(shape);
      
      // Aplicar softmax por fila (batch)
      for (size_t i = 0; i < shape[0]; ++i) {
        // Encontrar el máximo para estabilidad numérica
        T max_val = z(i, 0);
        for (size_t j = 1; j < shape[1]; ++j) {
          max_val = std::max(max_val, z(i, j));
        }
        
        // Calcular exponenciales y suma
        T sum_exp = T{0};
        for (size_t j = 0; j < shape[1]; ++j) {
          T exp_val = std::exp(z(i, j) - max_val);
          result(i, j) = exp_val;
          sum_exp += exp_val;
        }
        
        // Normalizar
        for (size_t j = 0; j < shape[1]; ++j) {
          result(i, j) /= sum_exp;
        }
      }
      
      last_output_ = result;
      return result;
    }
    
    Tensor<T,2> backward(const Tensor<T,2>& grad_output) override {
      const auto& shape = grad_output.shape();
      Tensor<T,2> grad_input(shape);
      
      // Derivada de softmax: s_i * (delta_ij - s_j)
      for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
          T grad = T{0};
          for (size_t k = 0; k < shape[1]; ++k) {
            T delta = (j == k) ? T{1} : T{0};
            grad += grad_output(i, k) * last_output_(i, j) * (delta - last_output_(i, k));
          }
          grad_input(i, j) = grad;
        }
      }
      
      return grad_input;
    }
  };

} // namespace utec::neural_network

#endif // UTEC_NEURAL_NETWORK_ACTIVATION_H
