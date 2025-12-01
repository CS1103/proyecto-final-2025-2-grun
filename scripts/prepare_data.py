#!/usr/bin/env python3
"""
Script para procesar datos históricos de acciones y generar conjuntos de entrenamiento y prueba
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import glob

def calculate_technical_indicators(df):
    """Calcular indicadores técnicos para los datos de acciones"""
    
    # Cambios de precio
    df['price_change_1d'] = df['Adj Close'].pct_change(1)
    df['price_change_3d'] = df['Adj Close'].pct_change(3)
    df['price_change_5d'] = df['Adj Close'].pct_change(5)
    
    # Medias móviles simples
    df['sma_5'] = df['Adj Close'].rolling(window=5).mean()
    df['sma_10'] = df['Adj Close'].rolling(window=10).mean()
    df['sma_20'] = df['Adj Close'].rolling(window=20).mean()
    
    # RSI (Relative Strength Index)
    def calculate_rsi(prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    df['rsi'] = calculate_rsi(df['Adj Close'])
    
    # Ratio de volumen
    df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(window=10).mean()
    
    # Volatilidad (desviación estándar de retornos)
    df['volatility'] = df['price_change_1d'].rolling(window=10).std() * np.sqrt(252)  # Anualizada
    
    # Momentum (rate of change)
    df['momentum'] = df['Adj Close'].pct_change(10)
    
    # Label binario: 1 si sube al día siguiente, 0 si baja
    df['target'] = (df['Adj Close'].shift(-1) > df['Adj Close']).astype(int)
    
    return df

def load_and_process_stock_data(data_dir):
    """Cargar y procesar todos los archivos de acciones"""
    
    all_data = []
    
    # Directorios de datos
    stock_files = glob.glob(os.path.join(data_dir, 'stocks', '*.csv'))
    etf_files = glob.glob(os.path.join(data_dir, 'etfs', '*.csv'))
    
    all_files = stock_files + etf_files
    
    for file_path in all_files:
        symbol = Path(file_path).stem  # Obtener nombre del archivo sin extensión
        
        try:
            # Cargar datos
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            df['Symbol'] = symbol
            
            # Calcular indicadores técnicos
            df = calculate_technical_indicators(df)
            
            # Eliminar filas con NaN (debido a ventanas deslizantes)
            df = df.dropna()
            
            # Eliminar la última fila (no tiene target porque es el futuro)
            df = df[:-1]
            
            print(f"Procesado {symbol}: {len(df)} muestras")
            
            all_data.append(df)
            
        except Exception as e:
            print(f"Error procesando {file_path}: {e}")
    
    # Combinar todos los datos
    combined_df = pd.concat(all_data, ignore_index=True)
    
    return combined_df

def create_feature_matrix(df):
    """Crear matriz de características para entrenamiento"""
    
    feature_columns = [
        'price_change_1d', 'price_change_3d', 'price_change_5d',
        'sma_5', 'sma_10', 'sma_20', 'rsi', 'volume_ratio', 
        'volatility', 'momentum'
    ]
    
    # Seleccionar características y normalizar
    X = df[feature_columns].copy()
    
    # Normalizar características (z-score)
    X = (X - X.mean()) / X.std()
    
    # Target
    y = df['target']
    
    # Información adicional
    info_columns = ['Date', 'Symbol', 'Adj Close']
    info_df = df[info_columns].copy()
    
    return X, y, info_df

def main():
    # Configuración
    data_dir = '/home/cristhian/Desktop/Progra3/proyecto3/proyecto-final-2025-2-grun/data'
    output_dir = '/home/cristhian/Desktop/Progra3/proyecto3/proyecto-final-2025-2-grun'
    
    print("=== Procesador de Datos de Acciones ===")
    print(f"Procesando datos desde: {data_dir}")
    
    # Cargar y procesar datos
    print("\n1. Cargando datos...")
    df = load_and_process_stock_data(data_dir)
    print(f"Total de muestras combinadas: {len(df)}")
    
    # Crear matrices de características
    print("\n2. Extrayendo características...")
    X, y, info = create_feature_matrix(df)
    print(f"Características extraídas: {X.shape[1]}")
    print(f"Muestras válidas: {len(X)}")
    
    # División temporal (usar fechas más recientes para test)
    print("\n3. Dividiendo datos...")
    
    # Ordenar por fecha para división temporal
    combined_data = pd.concat([X, y, info], axis=1)
    combined_data = combined_data.sort_values('Date')
    
    # Usar 80% para entrenamiento, 20% para prueba (división temporal)
    split_idx = int(0.8 * len(combined_data))
    
    train_data = combined_data.iloc[:split_idx]
    test_data = combined_data.iloc[split_idx:]
    
    print(f"Datos de entrenamiento: {len(train_data)} muestras")
    print(f"Datos de prueba: {len(test_data)} muestras")
    
    # Mostrar distribución de clases
    print(f"Entrenamiento - Sube: {train_data['target'].sum()} ({train_data['target'].mean():.2%})")
    print(f"Prueba - Sube: {test_data['target'].sum()} ({test_data['target'].mean():.2%})")
    
    # Guardar archivos
    print("\n4. Guardando archivos...")
    
    train_file = os.path.join(output_dir, 'data_training.csv')
    test_file = os.path.join(output_dir, 'data_test.csv')
    
    train_data.to_csv(train_file, index=False)
    test_data.to_csv(test_file, index=False)
    
    print(f"✓ Archivo de entrenamiento guardado: {train_file}")
    print(f"✓ Archivo de prueba guardado: {test_file}")
    
    # Estadísticas finales
    print("\n=== RESUMEN ===")
    print(f"Archivos procesados: {len(glob.glob(os.path.join(data_dir, '*', '*.csv')))} símbolos")
    print(f"Período: {combined_data['Date'].min().strftime('%Y-%m-%d')} a {combined_data['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Características: {', '.join(X.columns.tolist())}")
    
    # Información por símbolo
    print("\nMuestras por símbolo:")
    symbol_counts = combined_data['Symbol'].value_counts()
    for symbol, count in symbol_counts.items():
        print(f"  {symbol}: {count}")

if __name__ == "__main__":
    main()