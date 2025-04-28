# AQI Prediction App

This project is a web application built using Streamlit that predicts Air Quality Index (AQI) based on input features using a Long Short-Term Memory (LSTM) model.

## Project Structure

```
aqi-prediction-app
├── models
│   └── Best (1).h5          # Saved LSTM model for AQI prediction
├── src
│   ├── app.py                # Main entry point for the Streamlit application
│   └── utils
│       └── preprocess.py     # Utility functions for data preprocessing
├── requirements.txt          # List of dependencies
└── README.md                 # Project documentation
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd aqi-prediction-app
   ```

2. **Create a virtual environment (optional but recommended):**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages:**
   ```
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Streamlit application:**
   ```
   streamlit run src/app.py
   ```

2. **Access the application:**
   Open your web browser and go to `http://localhost:8501` to interact with the AQI prediction app.

## Features

- Input features for AQI prediction.
- Visualization of predicted AQI values.
- User-friendly interface for easy interaction.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.