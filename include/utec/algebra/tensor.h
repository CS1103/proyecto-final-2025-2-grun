#ifndef UTEC_ALGEBRA_TENSOR_H
#define UTEC_ALGEBRA_TENSOR_H

#include <vector>
#include <array>
#include <initializer_list>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <type_traits>

namespace utec::algebra {

template<typename T, size_t DIMS>
class Tensor {
private:
    std::vector<T> data_;
    std::array<size_t, DIMS> shape_;
    std::array<size_t, DIMS> strides_;

    void calculate_strides() {
        strides_[DIMS - 1] = 1;
        for (int i = DIMS - 2; i >= 0; --i) {
            strides_[i] = strides_[i + 1] * shape_[i + 1];
        }
    }

    size_t get_index(const std::array<size_t, DIMS>& indices) const {
        size_t index = 0;
        for (size_t i = 0; i < DIMS; ++i) {
            index += indices[i] * strides_[i];
        }
        return index;
    }

public:
    // Constructor con std::array
    explicit Tensor(const std::array<size_t, DIMS>& shape) : shape_(shape) {
        size_t total_size = 1;
        for (size_t dim : shape_) {
            total_size *= dim;
        }
        data_.resize(total_size);
        calculate_strides();
    }

    // Constructor específico para 2D con dos parámetros size_t
    template<size_t D = DIMS>
    Tensor(typename std::enable_if_t<D == 2, size_t> rows, size_t cols) 
        : shape_{{rows, cols}} {
        size_t total_size = rows * cols;
        data_.resize(total_size);
        calculate_strides();
    }

    // Constructor por defecto
    Tensor() {
        for (auto& dim : shape_) {
            dim = 0;
        }
        calculate_strides();
    }

    // Constructor con initializer_list
    Tensor(const std::array<size_t, DIMS>& shape, std::initializer_list<T> init_list) 
        : shape_(shape) {
        size_t total_size = 1;
        for (size_t dim : shape_) {
            total_size *= dim;
        }
        data_.resize(total_size);
        calculate_strides();
        
        size_t i = 0;
        for (const auto& val : init_list) {
            if (i < data_.size()) {
                data_[i++] = val;
            }
        }
    }

    // Operador de asignación con initializer_list
    Tensor& operator=(std::initializer_list<T> init_list) {
        size_t i = 0;
        for (const auto& val : init_list) {
            if (i < data_.size()) {
                data_[i++] = val;
            }
        }
        return *this;
    }

    // Acceso para 2D
    template<size_t D = DIMS>
    typename std::enable_if_t<D == 2, T&> operator()(size_t i, size_t j) {
        return data_[i * strides_[0] + j * strides_[1]];
    }

    template<size_t D = DIMS>
    typename std::enable_if_t<D == 2, const T&> operator()(size_t i, size_t j) const {
        return data_[i * strides_[0] + j * strides_[1]];
    }

    // Métodos básicos
    const std::array<size_t, DIMS>& shape() const { return shape_; }
    size_t size() const { return data_.size(); }
    
    void fill(const T& value) {
        std::fill(data_.begin(), data_.end(), value);
    }

    // Iteradores
    typename std::vector<T>::iterator begin() { return data_.begin(); }
    typename std::vector<T>::iterator end() { return data_.end(); }
    typename std::vector<T>::const_iterator begin() const { return data_.begin(); }
    typename std::vector<T>::const_iterator end() const { return data_.end(); }

    // Operadores aritméticos
    Tensor operator+(const Tensor& other) const {
        Tensor result = *this;
        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] += other.data_[i];
        }
        return result;
    }

    Tensor operator-(const Tensor& other) const {
        Tensor result = *this;
        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] -= other.data_[i];
        }
        return result;
    }

    Tensor operator*(const Tensor& other) const {
        Tensor result = *this;
        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] *= other.data_[i];
        }
        return result;
    }

    Tensor operator/(const T& scalar) const {
        Tensor result = *this;
        for (auto& val : result.data_) {
            val /= scalar;
        }
        return result;
    }

    // Producto matricial para tensores 2D
    template<size_t D = DIMS>
    typename std::enable_if_t<D == 2, Tensor> dot(const Tensor& other) const {
        size_t rows = shape_[0];
        size_t cols = other.shape_[1];
        size_t inner = shape_[1];
        
        Tensor result({rows, cols});
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                T sum = T{};
                for (size_t k = 0; k < inner; ++k) {
                    sum += (*this)(i, k) * other(k, j);
                }
                result(i, j) = sum;
            }
        }
        return result;
    }

    // Función transpose para tensores 2D
    template<size_t D = DIMS>
    typename std::enable_if_t<D == 2, Tensor> transpose() const {
        Tensor result({shape_[1], shape_[0]});
        for (size_t i = 0; i < shape_[0]; ++i) {
            for (size_t j = 0; j < shape_[1]; ++j) {
                result(j, i) = (*this)(i, j);
            }
        }
        return result;
    }
};

// Operador de salida
template<typename T, size_t DIMS>
std::ostream& operator<<(std::ostream& os, const Tensor<T, DIMS>& tensor) {
    if constexpr (DIMS == 2) {
        const auto& shape = tensor.shape();
        os << "{\n";
        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                if (j > 0) os << " ";
                T val = tensor(i, j);
                if (std::abs(val - std::round(val)) < T{1e-10} && val >= -1000 && val <= 1000) {
                    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
                        os << static_cast<int>(std::round(val)) << ".0";
                    } else {
                        os << static_cast<int>(std::round(val));
                    }
                } else {
                    os << val;
                }
            }
            if (i < shape[0] - 1) os << "\n";
        }
        os << "\n}";
    }
    return os;
}

template<typename T, size_t DIMS, typename Func>
auto apply(const Tensor<T, DIMS>& tensor, Func func) {
    using ReturnType = decltype(func(std::declval<T>()));
    Tensor<ReturnType, DIMS> result(tensor.shape());
    
    auto it_result = result.begin();
    for (auto it = tensor.begin(); it != tensor.end(); ++it, ++it_result) {
        *it_result = func(*it);
    }
    return result;
}

} // namespace utec::algebra

#endif // UTEC_ALGEBRA_TENSOR_H