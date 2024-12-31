from src.data.collector import OwlracleCollector
from src.models.moving_average import MovingAverageModel
from tabulate import tabulate
import pandas as pd
from datetime import datetime

def main():
    collector = OwlracleCollector()
    model = MovingAverageModel()
    
    try:
        print("🔍 Collecting data...")
        data = collector.collect_data()
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        summary_stats = [
            ['Latest Gas Price', f"{df['gas_price'].iloc[-1]:.2f} Gwei"],
            ['Median (24h)', f"{df['gas_price'].median():.2f} Gwei"],
            ['Min (24h)', f"{df['gas_price'].min():.2f} Gwei"],
            ['Max (24h)', f"{df['gas_price'].max():.2f} Gwei"],
            ['Volatility', f"{df['gas_price'].std():.2f} Gwei"]
        ]
        
        print("\n📊 Gas Price Summary:")
        print(tabulate(summary_stats, tablefmt='grid'))
        
        # Model evaluation with fixed interval
        train_size = int(len(df) * 0.8)
        train_data = df[:train_size]
        
        model.train(train_data)
        predictions = model.predict('1h')  # Pass the interval string instead of DataFrame
        
        metrics = model.evaluate(df[train_size:])
        
        print("\n📈 Model Performance:")
        metrics_table = [
            ['Mean Absolute Error', f"{metrics['mae']:.4f} Gwei"],
            ['Root Mean Squared Error', f"{metrics['rmse']:.4f} Gwei"]
        ]
        print(tabulate(metrics_table, tablefmt='grid'))
        
        print(f"\n⏰ Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main() 