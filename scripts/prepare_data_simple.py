#!/usr/bin/env python3
"""
Script simple para procesar datos históricos de acciones sin dependencias externas
"""

import csv
import os
import glob
from datetime import datetime, timedelta
import math

def parse_date(date_str):
    """Convertir string de fecha a objeto datetime"""
    return datetime.strptime(date_str, '%Y-%m-%d')

def calculate_sma(prices, window):
    """Calcular media móvil simple"""
    if len(prices) < window:
        return None
    return sum(prices[-window:]) / window

def calculate_rsi(prices, window=14):
    """Calcular RSI (Relative Strength Index)"""
    if len(prices) < window + 1:
        return 50.0  # Valor neutral
    
    gains, losses = [], []
    for i in range(1, len(prices)):
        change = prices[i] - prices[i-1]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(-change)
    
    if len(gains) < window:
        return 50.0
    
    avg_gain = sum(gains[-window:]) / window
    avg_loss = sum(losses[-window:]) / window
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_volatility(returns, window=10):
    """Calcular volatilidad"""
    if len(returns) < window:
        return 0.0
    
    recent_returns = returns[-window:]
    mean_return = sum(recent_returns) / len(recent_returns)
    variance = sum((r - mean_return) ** 2 for r in recent_returns) / len(recent_returns)
    return math.sqrt(variance) * math.sqrt(252)  # Anualizada

def process_stock_file(file_path, use_recent_data=True, recent_years=2, 
                      use_sampling=True, sampling_interval=3, max_samples=300):
    """Procesar un archivo CSV de acciones con filtros para reducir dataset"""
    symbol = os.path.basename(file_path).replace('.csv', '')
    
    rows = []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Convertir a números
                row['Open'] = float(row['Open'])
                row['High'] = float(row['High'])
                row['Low'] = float(row['Low'])
                row['Close'] = float(row['Close'])
                row['Adj Close'] = float(row['Adj Close'])
                row['Volume'] = float(row['Volume'])
                row['Date'] = parse_date(row['Date'])
                row['Symbol'] = symbol
                rows.append(row)
            except (ValueError, KeyError):
                continue
    
    # Ordenar por fecha
    rows.sort(key=lambda x: x['Date'])
    
    # === APLICAR FILTROS PARA REDUCIR DATASET ===
    original_count = len(rows)
    
    # Filtrar por fechas recientes
    if use_recent_data and rows:
        cutoff_date = datetime.now() - timedelta(days=365 * recent_years)
        rows = [r for r in rows if r['Date'] >= cutoff_date]
        print(f"    Después de filtrar últimos {recent_years} años: {len(rows)} registros")
    
    # Muestreo temporal
    if use_sampling and len(rows) > 0:
        rows = rows[::sampling_interval]  # Tomar cada N días
        print(f"    Después de muestreo cada {sampling_interval} días: {len(rows)} registros")
    
    # Limitar número máximo de muestras
    if max_samples and len(rows) > max_samples:
        rows = rows[-max_samples:]  # Tomar las últimas N muestras
        print(f"    Después de limitar a {max_samples} muestras: {len(rows)} registros")
    
    print(f"    Reducción: {original_count} -> {len(rows)} ({len(rows)/original_count:.1%} del original)")
    
    processed_data = []
    
    for i, row in enumerate(rows):
        if i < 25:  # Necesitamos al menos 25 días de historia
            continue
        
        # Obtener precios históricos
        prices = [r['Adj Close'] for r in rows[:i+1]]
        volumes = [r['Volume'] for r in rows[:i+1]]
        returns = []
        
        # Calcular retornos
        for j in range(1, len(prices)):
            if prices[j-1] != 0:
                returns.append((prices[j] - prices[j-1]) / prices[j-1])
        
        # Calcular características técnicas
        features = {}
        
        # Cambios de precio
        if len(prices) >= 2:
            features['price_change_1d'] = (prices[-1] - prices[-2]) / prices[-2] if prices[-2] != 0 else 0
        if len(prices) >= 4:
            features['price_change_3d'] = (prices[-1] - prices[-4]) / prices[-4] if prices[-4] != 0 else 0
        if len(prices) >= 6:
            features['price_change_5d'] = (prices[-1] - prices[-6]) / prices[-6] if prices[-6] != 0 else 0
        
        # Medias móviles
        features['sma_5'] = calculate_sma(prices, 5) or prices[-1]
        features['sma_10'] = calculate_sma(prices, 10) or prices[-1]
        features['sma_20'] = calculate_sma(prices, 20) or prices[-1]
        
        # Normalizar SMAs (ratio con precio actual)
        features['sma_5'] = features['sma_5'] / prices[-1] if prices[-1] != 0 else 1
        features['sma_10'] = features['sma_10'] / prices[-1] if prices[-1] != 0 else 1
        features['sma_20'] = features['sma_20'] / prices[-1] if prices[-1] != 0 else 1
        
        # RSI
        features['rsi'] = calculate_rsi(prices)
        
        # Ratio de volumen
        avg_volume = sum(volumes[-10:]) / min(10, len(volumes)) if volumes else 1
        features['volume_ratio'] = volumes[-1] / avg_volume if avg_volume != 0 else 1
        
        # Volatilidad
        features['volatility'] = calculate_volatility(returns)
        
        # Momentum
        if len(prices) >= 11:
            features['momentum'] = (prices[-1] - prices[-11]) / prices[-11] if prices[-11] != 0 else 0
        else:
            features['momentum'] = 0
        
        # Target (label): 1 si sube el día siguiente, 0 si baja
        target = 0
        if i < len(rows) - 1:
            next_price = rows[i + 1]['Adj Close']
            target = 1 if next_price > row['Adj Close'] else 0
        else:
            continue  # Saltar última fila (no hay precio siguiente)
        
        # Crear registro
        record = {
            'Date': row['Date'].strftime('%Y-%m-%d'),
            'Symbol': symbol,
            'Close': row['Adj Close'],
            **features,
            'target': target
        }
        
        processed_data.append(record)
    
    return processed_data

def normalize_features(data, feature_names):
    """Normalizar características usando z-score"""
    
    # Calcular estadísticas
    stats = {}
    for feature in feature_names:
        values = [row[feature] for row in data]
        mean_val = sum(values) / len(values)
        var_val = sum((v - mean_val) ** 2 for v in values) / len(values)
        std_val = math.sqrt(var_val) if var_val > 0 else 1
        stats[feature] = {'mean': mean_val, 'std': std_val}
    
    # Normalizar
    for row in data:
        for feature in feature_names:
            original = row[feature]
            mean_val = stats[feature]['mean']
            std_val = stats[feature]['std']
            row[feature] = (original - mean_val) / std_val if std_val > 0 else 0
    
    return data, stats

def main():
    # === CONFIGURACIÓN PARA REDUCIR DATASET ===
    USE_RECENT_DATA = False     # No filtrar por fecha (datos terminan en 2020)
    RECENT_YEARS = 5           # No usado si USE_RECENT_DATA = False
    
    USE_SAMPLING = True        # Muestreo temporal
    SAMPLING_INTERVAL = 5      # Cada 5 días (más reducción)
    
    MAX_SAMPLES_PER_STOCK = 500  # Máximo por acción (tomar más datos recientes)
    
    # Directorios
    data_dir = '/home/cristhian/Desktop/Progra3/proyecto3/proyecto-final-2025-2-grun/data'
    
    print("=== Procesador de Datos de Acciones para Stock Predictor ===")
    if USE_RECENT_DATA:
        print(f"Configuración: Últimos {RECENT_YEARS} años, cada {SAMPLING_INTERVAL} días, máx {MAX_SAMPLES_PER_STOCK} muestras/stock")
    else:
        print(f"Configuración: Todos los datos, cada {SAMPLING_INTERVAL} días, máx {MAX_SAMPLES_PER_STOCK} muestras/stock (más recientes)")
    
    # Solo procesar archivos de stocks (no ETFs)
    stock_files = glob.glob(os.path.join(data_dir, 'stocks', '*.csv'))
    
    print(f"Archivos de stocks encontrados: {len(stock_files)}")
    print("Stocks a procesar:")
    for f in stock_files:
        print(f"  - {os.path.basename(f)}")
    
    # Procesar solo archivos de stocks
    all_data = []
    for file_path in stock_files:
        try:
            print(f"Procesando {os.path.basename(file_path)}...")
            data = process_stock_file(file_path, USE_RECENT_DATA, RECENT_YEARS, 
                                    USE_SAMPLING, SAMPLING_INTERVAL, MAX_SAMPLES_PER_STOCK)
            all_data.extend(data)
            print(f"  -> {len(data)} muestras")
        except Exception as e:
            print(f"  Error: {e}")
    
    print(f"\nTotal muestras: {len(all_data)}")
    
    # Ordenar por fecha
    all_data.sort(key=lambda x: datetime.strptime(x['Date'], '%Y-%m-%d'))
    
    # Normalizar características
    feature_names = ['price_change_1d', 'price_change_3d', 'price_change_5d',
                     'sma_5', 'sma_10', 'sma_20', 'rsi', 'volume_ratio',
                     'volatility', 'momentum']
    
    print("Normalizando características...")
    all_data, norm_stats = normalize_features(all_data, feature_names)
    
    # Dividir datos (80% entrenamiento, 20% prueba)
    split_idx = int(0.8 * len(all_data))
    train_data = all_data[:split_idx]
    test_data = all_data[split_idx:]
    
    print(f"Entrenamiento: {len(train_data)} muestras")
    print(f"Prueba: {len(test_data)} muestras")
    
    # Estadísticas de target
    train_ups = sum(1 for row in train_data if row['target'] == 1)
    test_ups = sum(1 for row in test_data if row['target'] == 1)
    
    print(f"Train - Sube: {train_ups} ({train_ups/len(train_data):.2%})")
    print(f"Test - Sube: {test_ups} ({test_ups/len(test_data):.2%})")
    
    # Guardar archivos
    fieldnames = ['Date', 'Symbol', 'Close'] + feature_names + ['target']
    
    # Archivo de entrenamiento
    train_file = '/home/cristhian/Desktop/Progra3/proyecto3/proyecto-final-2025-2-grun/stock_data_training.csv'
    with open(train_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(train_data)
    
    # Archivo de prueba
    test_file = '/home/cristhian/Desktop/Progra3/proyecto3/proyecto-final-2025-2-grun/stock_data_test.csv'
    with open(test_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(test_data)
    
    print(f"\n✓ Archivos guardados:")
    print(f"  {train_file}")
    print(f"  {test_file}")
    
    # Mostrar estadísticas de normalización
    print(f"\nEstadísticas de normalización:")
    for feature in feature_names:
        mean_val = norm_stats[feature]['mean']
        std_val = norm_stats[feature]['std']
        print(f"  {feature}: mean={mean_val:.6f}, std={std_val:.6f}")

if __name__ == "__main__":
    main()