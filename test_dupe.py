import pandas as pd

path = "data/supplychain_cleaned.csv"
df = pd.read_csv(path, encoding='latin1')

reverse_map = {
    'type': 'Type',
    'days_for_shipping_(real)': 'Days for shipping (real)',
    'days_for_shipment_(scheduled)': 'Days for shipment (scheduled)',
    'delivery_status': 'Delivery Status',
    'late_delivery_risk': 'Late_delivery_risk',
    'category_id': 'Category Id',
    'category_name': 'Category Name',
    'customer_city': 'Customer City',
    'customer_country': 'Customer Country',
    'customer_segment': 'Customer Segment',
    'customer_state': 'Customer State',
    'customer_zipcode': 'Customer Zipcode',
    'department_id': 'Department Id',
    'department_name': 'Department Name',
    'market': 'Market',
    'order_city': 'Order City',
    'order_country': 'Order Country',
    'order_customer_id': 'Order Customer Id',
    'order_date': 'order date (DateOrders)',
    'order_id': 'Order Id',
    'order_item_discount': 'Order Item Discount',
    'order_item_discount_rate': 'Order Item Discount Rate',
    'order_item_profit_ratio': 'Order Item Profit Ratio',
    'order_item_quantity': 'Order Item Quantity',
    'sales': 'Sales',
    'order_item_total': 'Order Item Total',
    'order_profit_per_order': 'Order Profit Per Order',
    'order_region': 'Order Region',
    'order_state': 'Order State',
    'order_status': 'Order Status',
    'product_name': 'Product Name',
    'product_price': 'Product Price',
    'product_status': 'Product Status',
    'shipping_date': 'shipping date (DateOrders)',
    'shipping_mode': 'Shipping Mode'
}
df.rename(columns=reverse_map, inplace=True)

dummy_cols = [
    'Customer Lname', 'Product Description', 'Order Zipcode', 'customer_email', 
    'customer_password', 'customer_fname', 'customer_street', 'latitude', 
    'longitude', 'product_image', 'customer_id', 'order_item_cardprod_id', 
    'order_item_id', 'sales_per_customer', 'benefit_per_order', 'product_card_id', 
    'product_category_id', 'order_item_product_price', 'profit_flipped',
    'order_date_(dateorders)', 'shipping_date_(dateorders)'
]
for col in dummy_cols:
    if col not in df.columns:
        df[col] = "0"  

if 'order date (DateOrders)' in df.columns:
    df['order_date_(dateorders)'] = df['order date (DateOrders)']
if 'shipping date (DateOrders)' in df.columns:
    df['shipping_date_(dateorders)'] = df['shipping date (DateOrders)']

df.rename(columns={
    'order_date_(dateorders)': 'order_date',
    'shipping_date_(dateorders)': 'shipping_date'
}, inplace=True)

vc = df.columns.value_counts()
print("Duplicates:", vc[vc > 1])
