#ifndef UTEC_NEURAL_NETWORK_H
#define UTEC_NEURAL_NETWORK_H

#include "nn_interfaces.h"
#include "nn_dense.h"
#include "nn_activation.h"
#include "nn_loss.h"
#include "nn_optimizer.h"
#include <vector>
#include <memory>

namespace utec::neural_network {

template<typename T>
class NeuralNetwork {
private:
    std::vector<std::unique_ptr<ILayer<T>>> layers_;
    
public:
    void add_layer(std::unique_ptr<ILayer<T>> layer) {
        layers_.push_back(std::move(layer));
    }
    
    template <template <typename...> class LossType, 
              template <typename...> class OptimizerType = SGD>
    void train(const Tensor<T,2>& X, const Tensor<T,2>& Y, 
               const size_t epochs, const size_t batch_size, T learning_rate) {
        
        OptimizerType<T> optimizer(learning_rate);
        
        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            // Forward pass
            Tensor<T,2> current_input = X;
            
            // Propagar hacia adelante a través de todas las capas
            for (auto& layer : layers_) {
                current_input = layer->forward(current_input);
            }
            
            // Calcular pérdida
            LossType<T> loss(current_input, Y);
            
            // Backward pass
            Tensor<T,2> grad = loss.loss_gradient();
            
            // Propagar hacia atrás a través de todas las capas
            for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
                grad = (*it)->backward(grad);
            }
            
            // Actualizar parámetros
            for (auto& layer : layers_) {
                layer->update_params(optimizer);
            }
            optimizer.step();
        }
    }
    
    // Para realizar predicciones
    Tensor<T,2> predict(const Tensor<T,2>& X) {
        Tensor<T,2> current = X;
        
        // Forward pass a través de todas las capas
        for (auto& layer : layers_) {
            current = layer->forward(current);
        }
        
        return current;
    }
};

} // namespace utec::neural_network

#endif // UTEC_NEURAL_NETWORK_H
