#ifndef UTEC_NEURAL_NETWORK_DENSE_H
#define UTEC_NEURAL_NETWORK_DENSE_H

#include "nn_interfaces.h"

namespace utec::neural_network {

  template<typename T>
  class Dense final : public ILayer<T> {
  private:
    Tensor<T,2> weights_;
    Tensor<T,2> biases_;
    mutable Tensor<T,2> last_input_;
    mutable Tensor<T,2> dW_;
    mutable Tensor<T,2> db_;
    
  public:
    template<typename InitWFun, typename InitBFun>
    Dense(size_t in_features, size_t out_features, InitWFun init_w_fun, InitBFun init_b_fun) 
        : weights_(in_features, out_features), 
          biases_(1, out_features),
          last_input_(1, 1),
          dW_(in_features, out_features),
          db_(1, out_features) {
      
      // Inicializar pesos y sesgos usando las funciones proporcionadas
      init_w_fun(weights_);
      init_b_fun(biases_);
    }
    
    Tensor<T,2> forward(const Tensor<T,2>& x) override {
      last_input_ = x;

      const auto& x_shape = x.shape();
      const auto& w_shape = weights_.shape();
      size_t batch_size = x_shape[0];
      size_t out_features = w_shape[1];
      
      Tensor<T,2> result(batch_size, out_features);
      
      // Producto matricial: result = x * weights
      for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < out_features; ++j) {
          T sum = T{0};
          for (size_t k = 0; k < w_shape[0]; ++k) {
            sum += x(i, k) * weights_(k, j);
          }
          // Agregar bias
          result(i, j) = sum + biases_(0, j);
        }
      }
      
      return result;
    }
    
    Tensor<T,2> backward(const Tensor<T,2>& dZ) override {
      
      const auto& dZ_shape = dZ.shape();
      const auto& input_shape = last_input_.shape();
      size_t batch_size = dZ_shape[0];
      size_t out_features = dZ_shape[1];
      size_t in_features = input_shape[1];
      
      // Calcular dW = X^T * dZ
      for (size_t i = 0; i < in_features; ++i) {
        for (size_t j = 0; j < out_features; ++j) {
          T sum = T{0};
          for (size_t b = 0; b < batch_size; ++b) {
            sum += last_input_(b, i) * dZ(b, j);
          }
          dW_(i, j) = sum;
        }
      }
      
      // Calcular db = sum(dZ, axis=0)
      for (size_t j = 0; j < out_features; ++j) {
        T sum = T{0};
        for (size_t b = 0; b < batch_size; ++b) {
          sum += dZ(b, j);
        }
        db_(0, j) = sum;
      }
      
      // Calcular dX = dZ * W^T
      Tensor<T,2> dX(batch_size, in_features);
      for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < in_features; ++j) {
          T sum = T{0};
          for (size_t k = 0; k < out_features; ++k) {
            sum += dZ(i, k) * weights_(j, k);
          }
          dX(i, j) = sum;
        }
      }
      
      return dX;
    }
    
    void update_params(IOptimizer<T>& optimizer) override {
      // Actualizar pesos y sesgos usando el optimizador
      optimizer.update(weights_, dW_);
      optimizer.update(biases_, db_);
      optimizer.step();
    }
  };

} // namespace utec::neural_network

#endif // UTEC_NEURAL_NETWORK_DENSE_H
