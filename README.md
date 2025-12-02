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

**Algoritmos de entrenamiento: backpropagation, optimizadores.**

---

### 2. Diseño e implementación

#### 2.1 Arquitectura de la solución

* **Patrones de diseño**: ejemplo: Factory para capas, Strategy para optimizadores.
* **Estructura de carpetas (ejemplo)**:

  ```
  proyecto-final/
  ├── src/
  │   ├── layers/
  │   ├── optimizers/
  │   └── main.cpp
  ├── tests/
  └── docs/
  ```

#### 2.2 Manual de uso y casos de prueba

* **Cómo ejecutar**: `./build/neural_net_demo input.csv output.csv`
* **Casos de prueba**:

  * Test unitario de capa densa.
  * Test de función de activación ReLU.
  * Test de convergencia en dataset de ejemplo.

> *Personalizar rutas, comandos y casos reales.*

---

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
---

### Licencia

Este proyecto usa la licencia **MIT**. Ver [LICENSE](LICENSE) para detalles.

---
