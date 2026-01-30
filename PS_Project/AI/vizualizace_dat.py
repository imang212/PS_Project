import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def generuj_grafy(csv_path="log_detekce.csv"):
    try:
        # 1. Načtení dat
        df = pd.read_csv(csv_path)
        
        # Převod timestampu na čitelný čas
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 2. Agregace dat (počet detekcí po minutách)
        df.set_index('timestamp', inplace=True)
        stats = df['label'].resample('1Min').count()

        # 3. Vykreslení grafu
        plt.figure(figsize=(12, 6))
        plt.plot(stats.index, stats.values, marker='o', linestyle='-', color='b')
        
        plt.title('Intenzita provozu v čase (Demo Datacentrum)')
        plt.xlabel('Čas (minuty)')
        plt.ylabel('Počet detekovaných objektů')
        plt.grid(True)
        
        # Uložení grafu jako obrázek pro prezentaci
        plt.savefig('graf_provozu.png')
        print("[INFO] Graf byl uložen jako 'graf_provozu.png'")
        plt.show()

    except Exception as e:
        print(f"[ERROR] Nepodařilo se vygenerovat graf: {e}")

if __name__ == "__main__":
    generuj_grafy()