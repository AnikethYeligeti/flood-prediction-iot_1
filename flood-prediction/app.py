"""
IoT-Based Rainfall and Water-Level Data Preprocessing for Flood Prediction
Main Flask Application Entry Point
"""

from src.api import create_app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
