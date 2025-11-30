cmake_minimum_required(VERSION 3.18)
project(PortfolioOptimizer VERSION 1.0.0 LANGUAGES CXX)

# Configuración del estándar C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configuración de compilación
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Incluir directorios
include_directories(${CMAKE_SOURCE_DIR}/include)

# Crear biblioteca con las clases del optimizador de cartera
add_library(portfolio_optimizer STATIC
    src/utec/apps/portfolio_optimizer.cpp
    src/utec/apps/data_loader.cpp
)

target_include_directories(portfolio_optimizer PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# Crear ejecutable para el ejemplo
add_executable(portfolio_example
    src/utec/apps/portfolio_example.cpp
)

target_link_libraries(portfolio_example
    portfolio_optimizer
)

target_include_directories(portfolio_example PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Configuración de compilación específica
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(portfolio_optimizer PRIVATE DEBUG_MODE)
    target_compile_definitions(portfolio_example PRIVATE DEBUG_MODE)
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
install(TARGETS portfolio_example
    RUNTIME DESTINATION bin
)