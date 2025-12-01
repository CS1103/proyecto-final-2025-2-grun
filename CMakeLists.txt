cmake_minimum_required(VERSION 3.18)
project(StockPredictor VERSION 1.0.0 LANGUAGES CXX)

# Configuración del estándar C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configuración de compilación
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Incluir directorios
include_directories(${CMAKE_SOURCE_DIR}/include)

# Crear biblioteca con las clases del predictor de acciones
add_library(stock_predictor_lib STATIC
    src/utec/apps/stock_predictor.cpp
    src/utec/apps/data_loader.cpp
)

target_include_directories(stock_predictor_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# Ejecutable para validar el modelo
add_executable(model_validator
    src/utec/apps/model_validator.cpp
)

target_link_libraries(model_validator
    stock_predictor_lib
)

target_include_directories(model_validator PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Configuración de compilación específica
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(stock_predictor_lib PRIVATE DEBUG_MODE)
    target_compile_definitions(model_validator PRIVATE DEBUG_MODE)
endif()

# Opcional: agregar tests si existen
if(EXISTS "${CMAKE_SOURCE_DIR}/tests/CMakeLists.txt")
    enable_testing()
    add_subdirectory(tests)
endif()

# Información del build
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Source directory: ${CMAKE_SOURCE_DIR}")
message(STATUS "Binary directory: ${CMAKE_BINARY_DIR}")

# Opcional: instalar ejecutables
install(TARGETS model_validator
    RUNTIME DESTINATION bin
)