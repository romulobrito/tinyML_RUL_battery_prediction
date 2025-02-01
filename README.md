# Battery Remaining Useful Life (RUL) Prediction

This project implements a Machine Learning model for predicting the Remaining Useful Life (RUL) of batteries, using TinyML techniques for optimization and deployment on resource-constrained devices.

## 📊 About the Project

The system uses historical battery charge/discharge cycle data to predict how much useful life remains for a battery in use. The main features analyzed include:

- Discharge time
- Voltage drop
- Maximum and minimum voltages
- Charging times
- Charging efficiency

## 🛠️ Technologies Used

- Python 3.10
- PyTorch (base model and quantization)
- Pandas & NumPy (data processing)
- Scikit-learn (preprocessing and evaluation)
- Matplotlib & Seaborn (visualization)

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/romulobrito/tinyML_RUL_battery_prediction.git
cd tinyML_RUL_battery_prediction
```

2. Create and activate a virtual environment:
```bash
python -m venv venv_battery
source venv_battery/bin/activate  # Linux/Mac
# or
.\venv_battery\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

The project includes a Jupyter notebook (`tinyML_RULBattery.ipynb`) that demonstrates:

1. Data loading and preprocessing
2. Exploratory data analysis
3. Model training
4. Knowledge distillation optimization
5. Model quantization
6. Performance evaluation

## 📈 Results

The current model achieves:
- MSE (Mean Squared Error) of 0.0017 for the Teacher model
- MSE of 0.0026 for the Student model (quantized)
- Significant model size reduction after quantization

Detailed evaluation metrics include:
- Overall MSE, MAE, and R² scores
- Performance analysis by RUL ranges (0-100, 100-500, 500-1000)
- Comprehensive model comparison between Teacher and Student models

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the project
2. Create your feature branch (`git checkout -b feature/Feature`)
3. Commit your changes (`git commit -m 'Add: new feature'`)
4. Push to the branch (`git push origin feature/Feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License.

## 📧 Contact
Romulo Brito - [romulobrito@lenep.uenf.br](mailto:romulobrito@lenep.uenf.br)

Project Link: [https://github.com/romulobrito/tinyML_RUL_battery_prediction](https://github.com/romulobrito/tinyML_RUL_battery_prediction) 