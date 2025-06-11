import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import urllib.parse
from scipy import stats # For statistical tests (optional, but good to keep)


INPUT_FILENAME='/home/teena/Documents/all_in_one/Datascience/olist_abt_loaded_comprehensive.csv'
OUTPUT_FILENAME='/home/teena/Documents/all_in_one/Datascience/olist_abt_processed_output.csv'
# Ensure df_abt exists after loading
df_abt=None

try:
    if INPUT_FILENAME.endswith('.parquet'):
        df_abt=pd.read_parquet(INPUT_FILENAME)
    elif INPUT_FILENAME.endswith('.csv'):
        df_abt=pd.read_csv(INPUT_FILENAME)
    else:
        print(f"Unsupported file format: {INPUT_FILENAME}")
        exit()

    if df_abt is not None:
        print("Successfully loaded the data from {INPUT_FILENAME}")

except FileNotFoundError:
    print(f"Error:File not found:{INPUT_FILENAME}")
    exit()
except Exception as e:
    print(f"An error occured while loadding the data :{INPUT_FILENAME}")
    exit()

if df_abt is None:
    print(f"Error :Unable to load data from {INPUT_FILENAME}")
    exit()

df_processed=df_abt.copy()
print("data loaded and copied to df_processed")
print(df_processed.head())
print(df_processed.shape)
print(df_processed.info())
print(df_processed.describe())
print(df_processed.columns)
print(df_processed.describe(include='object'))

#dropping columns
cols_to_drop = [
    'avg_product_name_length',
    'avg_product_desc_length',
    'avg_product_photos_qty'
]
df_processed.drop(columns=cols_to_drop, inplace=True, errors='ignore') # errors='ignore' is safe
print(df_processed.columns)
print("Missing values processing")

#check for missing values
missing_values=df_processed.isnull().sum()
print(missing_values[missing_values>0])

print(f"shape before handling missing values:{df_processed.shape}")
df_processed.dropna(subset=['payment_type','payment_installments','payment_value','payment_count'],inplace=True)
print(f"shape after handling missing values:{df_processed.shape}")

median_weight=df_processed['avg_product_weight_g'].median()
median_volume=df_processed['avg_product_volume_cm3'].median()

df_processed['avg_product_weight_g'].fillna(median_weight,inplace=True)
df_processed['avg_product_volume_cm3'].fillna(median_volume,inplace=True)

print(f"shape after handling missing values:{df_processed.shape}")

df_processed['product_category_name_english'].fillna('Unknown',inplace=True)

print(f"shape after handling missing values:{df_processed.shape}")

#no of rows in each column of the datafram
print(f"No of non null count in each columns:{df_processed.count()}")

#check for duplicates
print(f"No of duplicates:{df_processed.duplicated().sum()}")
#If you want to remve the duplicates
# df_processed.drop_duplicates(inplace=True)

if 'order_id' in df_processed.columns:
    order_id_duplicates=df_processed.duplicated(subset=['order_id']).sum()
    print(f"no of duplicates in order_id:{order_id_duplicates}")
# Find which order_ids are duplicated
duplicate_mask = df_processed.duplicated(subset=['order_id'], keep=False) # Mark ALL rows involved in duplication
duplicated_orders_df = df_processed[duplicate_mask].sort_values('order_id')

print("\n--- Sample of Rows with Duplicate order_id ---")
print(duplicated_orders_df[['order_id', 'review_id', 'review_score', 'review_creation_date']].head(20))

# Check if review_id is unique for these duplicated orders
print("\nIs review_id unique within the duplicated orders subset?")
print(duplicated_orders_df['review_id'].is_unique)
print(df_processed.columns)


#lets now remove duplicates and keep only the first review for each order_id

if 'order_id' in df_processed.columns and 'review_creation_date' in df_processed.columns:
    df_processed['review_creation_date']=pd.to_datetime(df_processed['review_creation_date'],errors='coerce')
    df_processed=df_processed.dropna(subset=['review_creation_date'])
    df_processed=df_processed.sort_values(['order_id','review_creation_date'],ascending=[True,False])
    df_processed=df_processed.drop_duplicates(subset=['order_id'],keep='first')
    print(f"shape after removing duplicates:{df_processed.shape}")

else:
    print("skipping removal of duplicates")

print(f"no of non nulls counts in each columns:{df_processed.count()}")


cols_to_check = [
    'delivery_days', 'delivery_diff_days', 'approval_hours', 'processing_hours',
    'carrier_hours', 'review_response_hours', 'payment_value', 'total_price',
    'total_freight', 'avg_product_weight_g', 'avg_product_volume_cm3'
]

cols=[col for col in cols_to_check if col in df_processed.columns]
print(cols)
outlier_summary={}
for col in cols:

    if not pd.api.types.is_numeric_dtype(df_processed[col]):
        continue
    Q1=df_processed[col].quantile(0.25)
    Q3=df_processed[col].quantile(0.75)
    IQR=Q3-Q1
    if IQR==0: #then all values are same
        continue

    lower_bound=Q1-2.5*IQR
    upper_bound=Q3+2.5*IQR

    is_outlier=(df_processed[col]<lower_bound) | (df_processed[col]>upper_bound)
    outlier_count=is_outlier.sum()
    outlier_summary[col]=outlier_count


for col,count in outlier_summary.items():
    print(f"{col}:{count} outliers")


df_processed.to_csv(OUTPUT_FILENAME, index=False)
print("successfully saved the processed data to {OUTPUT_FILENAME}")