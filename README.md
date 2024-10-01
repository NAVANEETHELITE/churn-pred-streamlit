# Customer Churn Prediction

This application predicts customer churn using a simple ANN model which was trained using 10000 rows of customer data.

## Project Structure

- `model.h5`: The trained ANN model used for prediction.
- `label_encoder_gender.pkl`: Pickled `LabelEncoder` for the 'Gender' column.
- `onehot_encoder_geo.pkl`: Pickled `OneHotEncoder` for the 'Geography' column.
- `scaler.pkl`: Pickled `StandardScaler` for scaling numerical input data.
- `app.py`: The main Streamlit application file where the prediction logic is implemented.

## Prerequisites

Before running the project, ensure you have the following installed:

- Python 3.8 or above
- Required Python libraries: 
  - Streamlit
  - TensorFlow
  - Scikit-learn
  - Pandas
  - Numpy

Libraries Installation:

```bash
pip install -r requirements.txt
```

Running the streamlit app:

```bash
streamlit run app.py
```
