I notice the previous response was cut off. Let me complete the documentation
for the `train_models.py` script:

### **💾 Model Persistence & Management**

```python
# Automatic model saving with metadata
model_filename = f"{symbol}_{model_type}_{timestamp}.joblib"
# Example: TCS_random_forest_20241215_143022.joblib

# Model directory structure
models/saved_models/
├── TCS_random_forest_20241215_143022.joblib
├── RELIANCE_gradient_boost_20241215_144521.joblib
├── INFY_random_forest_20241215_145123.joblib
└── ...

# Model metadata stored with each file:
- Symbol and model type
- Training timestamp
- Performance metrics
- Feature importance
- Training parameters
```

### **📊 Performance Monitoring**

```python
# Training statistics tracking
training_stats = {
    'total_trained': 0,
    'successful': 0,
    'failed': 0,
    'start_time': datetime,
    'end_time': datetime
}

# Performance evaluation criteria
evaluation = {
    'meets_threshold': bool,
    'meets_accuracy': accuracy >= 60%,
    'meets_r2': r2_score >= 0.5,
    'within_time_limit': time <= 300s,
    'performance_grade': 'A+' to 'D'
}
```

### **🔄 Parallel Processing**

```python
# ThreadPoolExecutor for concurrent training
max_workers = min(4, len(stock_list))  # Limit concurrent training

# Progress tracking for parallel execution
with Progress() as progress:
    task = progress.add_task("Training models...", total=len(stock_list))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all training tasks
        future_to_symbol = {executor.submit(...): symbol for symbol in stock_list}

        # Collect results as they complete
        for future in as_completed(future_to_symbol):
            # Handle individual results
```

### **🎯 Advanced Features**

#### **Cross-Validation**

```python
# K-fold cross-validation for model validation
cv_scores = cross_val_score(
    model.model, X_scaled, y,
    cv=5,  # 5-fold CV
    scoring='r2',
    n_jobs=-1  # Use all CPU cores
)
```

#### **Hyperparameter Optimization**

```python
# Grid search for optimal parameters
param_grids = {
    'random_forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5, 10]
    },
    'gradient_boost': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2]
    }
}
```

#### **Feature Importance Analysis**

```python
# Extract and rank feature importance
feature_importance = dict(zip(
    feature_columns,
    model.feature_importances_
))

# Sort by importance
top_features = sorted(
    feature_importance.items(),
    key=lambda x: x[1],
    reverse=True
)[:10]
```

### **📈 Results Export & Analysis**

```python
# JSON export with comprehensive metadata
json_data = {
    'export_timestamp': datetime.now().isoformat(),
    'export_version': '1.0',
    'results': results,
    'summary': {
        'total_results': len(results),
        'successful_results': success_count,
        'failed_results': failure_count,
        'average_accuracy': avg_accuracy,
        'performance_distribution': grade_counts
    }
}
```

### **🛠️ Configuration Management**

```python
class TrainingConfig:
    def __init__(self):
        self.models_dir = Path("models/saved_models")
        self.results_dir = Path("training_results")
        self.logs_dir = Path("logs/training")

        # Training parameters
        self.default_period = "2y"
        self.default_model = "random_forest"
        self.test_size = 0.2
        self.cv_folds = 5
        self.n_jobs = -1
        self.random_state = 42

        # Performance thresholds
        self.min_accuracy = 60.0
        self.min_r2_score = 0.5
        self.max_training_time = 300
```

### **🧹 Maintenance Features**

```bash
# Clean up old model files
poetry run train-models cleanup-models --older-than-days 30 --confirm

# List saved models with filtering
poetry run train-models list-models --symbol TCS --model-type random_forest

# Benchmark performance across models
poetry run train-models benchmark --symbol RELIANCE --iterations 5
```

### **📝 Logging & Monitoring**

```python
# Structured logging with rotation
logger.add(
    "logs/training/training_{time}.log",
    rotation="1 day",
    retention="30 days",
    level="INFO"
)

# Performance metrics tracking
- Training times per stock/model
- Success/failure rates
- Accuracy distributions
- Resource utilization
- Error patterns
```

### **🎨 Integration with Main Pipeline**

```python
# Import in main application
from scripts.train_models import ModelTrainer

# Use in API endpoints
trainer = ModelTrainer()
result = trainer.train_single_stock(
    symbol="TCS",
    model_type="random_forest",
    period="2y",
    save_model=True
)

# Background training tasks
background_tasks.add_task(
    trainer.train_single_stock,
    symbol, model_type, period
)
```

This comprehensive training script provides everything needed for professional-grade model training,
from single stock analysis to large-scale batch processing with performance monitoring,
automated evaluation, and intelligent model management! 🚀

The script integrates seamlessly with your Poetry-based pipeline and provides both
CLI and programmatic interfaces for maximum flexibility.
