import joblib
import pandas as pd
import sys

sys.path.insert(0, 'src')
from preprocess import predict_classifier

model = joblib.load('artifacts/classifier_model.pkl')
cols = joblib.load('artifacts/classifier_columns.pkl')

def test_combo(shipping_mode, days):
    data = pd.DataFrame([{
        'type': 'transfer', 
        'days_for_shipment_(scheduled)': days, 
        'customer_segment': 'consumer', 
        'department_name': 'fitness', 
        'category_name': 'cleats', 
        'shipping_mode': shipping_mode, 
        'market': 'LATAM', 
        'order_region': 'South America', 
        'product_price': 500.0, 
        'order_item_quantity': 3.0, 
        'sales': 1500.0
    }])
    pred, prob = predict_classifier(data, model, cols)
    return prob[0]

print("Standard Class, 1 day:", test_combo("standard class", 1.0))
print("Standard Class, 2 days:", test_combo("standard class", 2.0))
print("Standard Class, 4 days:", test_combo("standard class", 4.0))
print("First Class, 1 day:", test_combo("first class", 1.0))
print("Second Class, 2 days:", test_combo("second class", 2.0))
