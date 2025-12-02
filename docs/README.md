[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/o8XztwuW)
# Proyecto Final 2025-1: Predictor de Subida o Bajada de Acciones
## **CS2013 Programación III** · Informe Final

### **Descripción**

> Implementación de un algoritmo de predicción binaria usando redes neuronales en C++ para determinar si una acción subirá o bajará basándose en datos históricos y análisis técnico.

### Contenidos

1. [Datos generales](#datos-generales)
2. [Requisitos e instalación](#requisitos-e-instalación)
3. [Investigación teórica](#1-investigación-teórica)
4. [Diseño e implementación](#2-diseño-e-implementación)
5. [Ejecución](#3-ejecución)
6. [Análisis del rendimiento](#4-análisis-del-rendimiento)
7. [Trabajo en equipo](#5-trabajo-en-equipo)
8. [Conclusiones](#6-conclusiones)
9. [Bibliografía](#7-bibliografía)
10. [Licencia](#licencia)
---

### Datos generales

* **Tema**: Uso de redes neuronales para predecir el movimiento de acciones en la bolsa.
* **Grupo**: `grun`
* **Integrantes**:

  * Cristhian Jaimes Gamboa - 202120670

---

### Requisitos e instalación

1. **Compilador**: GCC con soporte para C++17 o superior
2. **Dependencias**:

   * CMake 3.18+
   * Biblioteca estándar de C++ (STL)
3. **Instalación**:

   ```bash
   git clone https://github.com/CS1103/proyecto-final-2025-2-grun.git
   cd proyecto-final-2025-2-grun
   mkdir build && cd build
   cmake ..
   make
   ```

4. **Ejecutables generados**:
   * `model_validator`: Validador del modelo de predicción de acciones

---

### 1. Investigación teórica

  **Evolución de las redes neuronales**

  El desarrollo de las redes neuronales se originó con los primeros estudios dedicados a descifrar el funcionamiento del sistema nervioso. Esta inspiración biológica permitió la creación de sistemas artificiales, conocidos hoy como Redes Neuronales Artificiales (RNA), las cuales han alcanzado un grado de sofisticación que les permite abordar y solucionar eficientemente problemas prácticos y reales. [1]
  1. Optimización Inicial y Computación Evolutiva
  El campo experimentó un avance significativo en sus etapas iniciales, especialmente desde finales de la década de 1980, con la introducción de la computación evolutiva. Este enfoque permitió la optimización de elementos cruciales de las redes, como sus arquitecturas, los pesos de conexión y las reglas de aprendizaje. La capacidad de utilizar métodos de búsqueda metaheurística facilitó que las RNA superaran las limitaciones de los diseños sencillos y comenzaran a abordar problemas de mayor complejidad. [2], [3]

  2. La Revolución del Deep Learning
  El hito más transformador fue el auge del Deep Learning. Las Redes Neuronales Profundas (DNNs) lograron un éxito sin precedentes en áreas como el reconocimiento de patrones y el aprendizaje automático, en gran medida impulsadas por el perfeccionamiento de algoritmos de entrenamiento como la retropropagación (backpropagation). Este avance metodológico permitió entrenar eficazmente redes con múltiples capas ocultas, desbloqueando capacidades que definen el panorama de la Inteligencia Artificial moderna. [4]

  3. Automatización del Diseño (Neuroevolución)
  La tendencia evolutiva más reciente se centra en la Neuroevolución (el uso de algoritmos evolutivos para el diseño de RNA) [5], [6]. Esta metodología ha ganado importancia al proporcionar un mecanismo eficiente para la Búsqueda Automatizada de Arquitecturas Neuronales (NAS) [7], [8]. La neuroevolución ofrece una alternativa poderosa a los métodos tradicionales basados en gradientes, permitiendo la exploración de arquitecturas, hiperparámetros y algoritmos de aprendizaje de manera paralela y a gran escala, lo que impulsa la eficiencia y el diseño de sistemas de Deep Learning de última generación.

**Algoritmos de entrenamiento: backpropagation y optimizadores**

1. Backpropagation (Propagación Inversa) - El backpropagation (propagación inversa) es el algoritmo fundamental y núcleo para entrenar redes neuronales. Su objetivo es ajustar los pesos de la red utilizando el método de descenso de gradiente para minimizar el error entre la salida predicha y la salida real. Se utiliza ampliamente en áreas como el reconocimiento de imágenes y el procesamiento del lenguaje natural. [9]

2. Desafíos y Optimizadores Tradicionales - Los optimizadores tradicionales, como el Descenso de Gradiente Estocástico (SGD), a menudo presentan inconvenientes, incluyendo una convergencia lenta y una alta sensibilidad a la elección de hiperparámetros [10].

3. Optimizadores Avanzados (Adaptativos e Híbridos) - Para superar estos desafíos y mejorar la eficiencia del entrenamiento, se han desarrollado métodos de optimización avanzados:

- Algoritmos Adaptativos: Optimizadores como el Gradiente Conjugado Estocástico Adaptativo (ASCG) mejoran la eficiencia al ajustar dinámicamente las tasas de aprendizaje y las direcciones de búsqueda. Esto conduce a una convergencia más rápida y una mejor capacidad de generalización. [10]

- Enfoques Híbridos: Estos combinan el backpropagation con métodos de computación evolutiva o basados en enjambres, como la Optimización por Enjambre de Partículas (PSO) y los Algoritmos Genéticos (GA). Estos híbridos han demostrado una mayor precisión y una convergencia más rápida en comparación con optimizadores estándar como Adam o SGD. [11], [12]

4. Innovaciones Recientes
Las innovaciones recientes buscan mejorar aún más las propiedades de convergencia y la eficiencia, incluso bajo restricciones de recursos. Esto incluye métodos como el descenso de gradiente de orden fraccional y el backpropagation espectral dinámico [13], [14].
---

### 2. Diseño e implementación

#### 2.1 Arquitectura de la solución

##### Patrones de diseño implementados

1. **Strategy Pattern (Patrón Estrategia)**:
   - **Optimizadores**: La interfaz `IOptimizer<T>` permite intercambiar algoritmos de optimización (SGD, Adam) sin modificar el código de las capas o la red neuronal.
   - **Funciones de pérdida**: La interfaz `ILoss<T, DIMS>` permite cambiar entre diferentes funciones de pérdida (MSELoss, BCELoss) de manera transparente.
   - **Capas de activación**: La interfaz `ILayer<T>` unifica el comportamiento de diferentes capas (Dense, ReLU, Sigmoid, Softmax), permitiendo que la red neuronal las trate de forma polimórfica.

2. **Template Method Pattern (Patrón Método Plantilla)**:
   - La clase `NeuralNetwork` define el flujo de entrenamiento (`train`) que siempre sigue los mismos pasos (forward → loss → backward → update), pero delega la implementación específica a las capas individuales.

3. **Composite Pattern (Patrón Compuesto)**:
   - `NeuralNetwork` actúa como contenedor de capas (`ILayer`), permitiendo construir arquitecturas complejas mediante composición de capas más simples.

##### Estructura del proyecto

```
proyecto-final-2025-2-grun/
├── include/utec/
│   ├── algebra/
│   │   └── tensor.h                    # Implementación de Tensor N-dimensional
│   ├── nn/
│   │   ├── nn_interfaces.h             # Interfaces: ILayer, IOptimizer, ILoss
│   │   ├── nn_dense.h                  # Capa Dense (fully connected)
│   │   ├── nn_activation.h             # Activaciones: ReLU, Sigmoid, Softmax
│   │   ├── nn_loss.h                   # Funciones de pérdida: MSE, BCE
│   │   ├── nn_optimizer.h              # Optimizadores: SGD, Adam
│   │   └── neural_network.h            # Clase principal NeuralNetwork
│   └── apps/
│       ├── data_loader.h               # Carga de datos desde CSV
│       └── stock_predictor.h           # Predictor de acciones
├── src/utec/apps/
│   ├── data_loader.cpp                 # Implementación del cargador de datos
│   ├── stock_predictor.cpp             # Extracción de features técnicas
│   └── model_validator.cpp             # Ejecutable de validación
├── tests/
│   ├── relu/test_[1-4]/                # Tests de función de activación ReLU
│   ├── dense/test_[1-4]/               # Tests de capa Dense
│   └── convergence/test_[1-4]/         # Tests de convergencia (XOR)
├── data/
│   ├── stocks/                         # Datos históricos de acciones (AAPL, GOOGL, JPM, JNJ, MSFT)
│   └── etfs/                           # Datos históricos de ETFs (SPY, VTI, QQQ, GLD, BND)
├── scripts/
│   └── prepare_data_simple.py          # Script de procesamiento de datos
├── build/                              # Directorio de compilación (generado)
├── stock_data_training.csv             # Dataset de entrenamiento (generado)
├── stock_data_test.csv                 # Dataset de prueba (generado)
├── CMakeLists.txt                      # Configuración de CMake
└── docs/
    └── README.md                       # Este documento
```

##### Componentes principales

1. **Capa de Álgebra**:
   - `Tensor<T, DIMS>`: Contenedor genérico N-dimensional que soporta operaciones matriciales fundamentales.

2. **Capa de Red Neuronal** (`include/utec/nn/`):
   - **Interfaces**: Define contratos para capas, optimizadores y funciones de pérdida.
   - **Dense Layer**: Implementa capa completamente conectada con inicializaciones He y Xavier.
   - **Activaciones**: ReLU, Sigmoid y Softmax con forward y backward pass.
   - **Funciones de pérdida**: MSE (regresión) y BCE (clasificación binaria).
   - **Optimizadores**: SGD básico y Adam con momentos adaptativos.

3. **Capa de Aplicación** (`include/utec/apps/`):
   - **DataLoader**: Parsea archivos CSV con datos de mercado (fecha, apertura, cierre, volumen).
   - **StockPredictor**: Extrae 10 características técnicas (SMA, RSI, volatilidad, momentum) y entrena modelo de clasificación binaria para predecir subida/bajada de acciones.

4. **Scripts de Procesamiento** (`scripts/`):
   - **prepare_data_simple.py**: Script Python independiente (sin dependencias externas) que:
     - Carga datos históricos desde `data/stocks/` (5 acciones: AAPL, GOOGL, JPM, JNJ, MSFT)
     - Calcula características técnicas: cambios de precio (1d, 3d, 5d), medias móviles (SMA 5, 10, 20), RSI, volatilidad, momentum, ratio de volumen
     - Genera labels binarios (1=sube, 0=baja) según el precio del día siguiente
     - Normaliza features usando z-score (media=0, desviación estándar=1)
     - Aplica muestreo temporal (cada 5 días) y limita a 500 muestras por acción para reducir el dataset
     - Divide datos en entrenamiento (80%) y prueba (20%)
     - Genera `stock_data_training.csv` y `stock_data_test.csv`

##### Características técnicas extraídas

El predictor calcula las siguientes 10 features por ventana temporal:
- `price_change_1d`, `price_change_3d`, `price_change_5d`: Cambios de precio a corto plazo
- `sma_5`, `sma_10`, `sma_20`: Medias móviles simples
- `rsi`: Índice de Fuerza Relativa (overbought/oversold)
- `volume_ratio`: Ratio del volumen vs promedio histórico
- `volatility`: Desviación estándar de retornos (10 días)
- `momentum`: Tasa de cambio de precio (10 días)

#### 2.2 Manual de uso y casos de prueba

* **Cómo ejecutar**: `./build/model_validator`

##### Tests Unitarios

El proyecto incluye 12 tests unitarios organizados en 3 categorías que validan el correcto funcionamiento de los componentes de la red neuronal:

**Compilación de la infraestructura base:**
```bash
make -C tests catch-essential
```

**Ejecución de tests individuales:**
```bash
# Ejemplo: ejecutar test de ReLU #1
cd tests/relu/test_1 && ./run_test

# Ejemplo: ejecutar test de Dense #2
cd tests/dense/test_2 && ./run_test

# Ejemplo: ejecutar test de Convergence #1
cd tests/convergence/test_1 && ./run_test
```

**Descripción de tests:**

| Categoría | Test | Nombre | Descripción |
|-----------|------|--------|-------------|
| **ReLU** | 1 | ReLU Forward-Backward Simple | Valida forward pass con valores mixtos (-1,2,0,-3) y backward con gradientes |
| **ReLU** | 2 | ReLU Diagonal Pattern | Matriz 5x4 con patrón diagonal, verifica ceros en negativos |
| **ReLU** | 3 | Sigmoid Forward-Backward | Compara función Sigmoid con valores extremos (±100) |
| **ReLU** | 4 | ReLU Gradient Validation | Verifica gradientes en backward pass (1 si x>0, 0 si x≤0) |
| **Dense** | 1 | Dense Forward Identity Init | Forward con inicialización identidad, verifica Y=X |
| **Dense** | 2 | Dense Backward Iota | Backward con datos secuenciales usando std::iota |
| **Dense** | 3 | Dense He Initialization | Inicialización He con seed=42, verifica forward/backward |
| **Dense** | 4 | Dense Xavier Initialization | Inicialización Xavier con seed=4, verifica forward/backward |
| **Convergence** | 1 | XOR MSELoss ReLU | Red 2→4→1 con ReLU, MSELoss, 3000 epochs, lr=0.08, seed=42 |
| **Convergence** | 2 | XOR BCELoss Sigmoid | Red 2→4→1 con Sigmoid, BCELoss, 4000 epochs, lr=0.08, seed=4 |
| **Convergence** | 3 | XOR MSELoss ReLU Low LR | Red 2→4→1 con ReLU, MSELoss, 4000 epochs, lr=0.02, seed=20 |
| **Convergence** | 4 | XOR BCELoss Sigmoid Alt Order | Red 2→4→1 con Sigmoid, BCELoss, 4000 epochs, lr=0.08, seed=4 |

**Criterios de éxito:**
- Tests de ReLU/Dense: Verifican dimensiones correctas y valores numéricos con tolerancia `epsilon(1e-12)`
- Tests de Convergence: Verifican que la red aprenda XOR correctamente (predicciones ≥0.6 para 1, <0.5 para 0)

### 3. Ejecución

> **Demo de ejemplo**: Video/demo alojado en `docs/demo.mp4`.
> Pasos:
>
> 1. Preparar datos de entrenamiento (formato CSV).
> 2. Ejecutar comando de entrenamiento.
> 3. Evaluar resultados con script de validación.

---

### 4. Análisis del rendimiento

* **Métricas de ejemplo**:

  * Iteraciones: 1000 épocas.
  * Tiempo total de entrenamiento: 2m30s.
  * Precisión final: 92.5%.
* **Ventajas/Desventajas**:

  * * Código ligero y dependencias mínimas.
  * – Sin paralelización, rendimiento limitado.
* **Mejoras futuras**:

  * Uso de BLAS para multiplicaciones (Justificación).
  * Paralelizar entrenamiento por lotes (Justificación).

---

### 5. Trabajo en equipo

| Tarea   | Miembro     | Rol                       |
| --------| --------    | ------------------------- |
| Todo    | Cristhian J. | Todo |
---

### 6. Conclusiones

* **Logros**: Implementar NN desde cero, validar en dataset de ejemplo.
* **Evaluación**: Calidad y rendimiento adecuados para propósito académico.
* **Aprendizajes**: Profundización en backpropagation y optimización.
* **Recomendaciones**: Escalar a datasets más grandes y optimizar memoria.

---

### 7. Bibliografía

- [1] A. Prieto, B. Prieto, E. Ortigosa, E. Ros, F. Pelayo, J. Ortega, and I. Rojas, "Neural networks: An overview of early research, current frameworks and new challenges," Neurocomputing, vol. 214, pp. 242–268, 2016. DOI: 10.1016/j.neucom.2016.06.014.
- [2] H. Ünal and F. Başçiftçi, "Evolutionary design of neural network architectures: a review of three decades of research," Artificial Intelligence Review, vol. 55, pp. 1723–1802, 2021. DOI: 10.1007/s10462-021-10049-5.
- [3] X. Yao, "Evolutionary Artificial Neural Networks," International Journal of Neural Systems, vol. 4, no. 3, pp. 203–222, 1993. DOI: 10.1142/s0129065793000171.
- [4] J. Schmidhuber, "Deep learning in neural networks: An overview," Neural Netw.: Off. J. Int. Neural Netw. Soc., vol. 61, pp. 85–117, 2014. DOI: 10.1016/j.neunet.2014.09.003.
- [5] A. Baldominos, Y. Sáez, and P. Isasi, "On the automated, evolutionary design of neural networks: past, present, and future," Neural Comput. Appl., vol. 32, pp. 519–545, 2019. DOI: 10.1007/s00521-019-04160-6.
- [6] K. Stanley, J. Clune, J. Lehman, and R. Miikkulainen, "Designing neural networks through neuroevolution," Nat. Mach. Intell., vol. 1, pp. 24–35, 2019. DOI: 10.1038/s42256-018-0006-z.
- [7] Y. Ma and Y. Xie, "Evolutionary neural networks for deep learning: a review," Int. J. Mach. Learn. Cybern., vol. 13, pp. 3001–3018, 2022. DOI: 10.1007/s13042-022-01578-8.
- [8] X. Zhou, A. Qin, M. Gong, and K. Tan, "A Survey on Evolutionary Construction of Deep Neural Networks," IEEE Trans. Evol. Comput., vol. 25, pp. 894–912, 2021. DOI: 10.1109/tevc.2021.3079985.
- [9] M. Li, "Comprehensive Review of Backpropagation Neural Networks," Acad. J. Sci. Technol., 2024. DOI: 10.54097/51y16r47.
- [10] I. Hashem, F. Alaba, M. Jumare, A. Ibrahim, and A. Abulfaraj, "Adaptive Stochastic Conjugate Gradient Optimization for Backpropagation Neural Networks," IEEE Access, vol. 12, pp. 33757–33768, 2024. DOI: 10.1109/access.2024.3370859.
- [11] S. Essang, S. Okeke, J. Effiong, R. Francis, S. Fadugba, A. Otobi, J. Auta, C. Chukwuka, M. Ogar-Abang, and A. Moses, "Adaptive hybrid optimization for backpropagation neural networks in image classification," in Proc. Nigerian Soc. Phys. Sci., 2025. DOI: 10.61298/pnspsc.2025.2.150.
- [12] A. Hazrati, S. Kariuki, and R. Silva, "Comparative Analysis of Backpropagation and Genetic Algorithms in Neural Network Training," Int. J. Health Technol. Innov., 2024. DOI: 10.60142/ijhti.v3i03.04.
- [13] C. Bao, Y. Pu, and Y. Zhang, "Fractional-Order Deep Backpropagation Neural Network," Comput. Intell. Neurosci., 2018. DOI: 10.1155/2018/7361628.
- [14] M. Muthuraman, "Dynamic Spectral Backpropagation for Efficient Neural Network Training," arXiv:2505.23369, 2025. [Online]. Available: https://arxiv.org/abs/2505.23369
---

### Licencia

Este proyecto usa la licencia **MIT**. Ver [LICENSE](LICENSE) para detalles.

---
