import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import urllib.parse
from scipy import stats # For statistical tests (optional, but good to keep)


df_processed=pd.read_csv('/home/teena/Documents/all_in_one/Datascience/olist_abt_processed_output.csv')

print(df_processed.head())
print(df_processed.shape)
print(df_processed.info())
print(df_processed.describe())
print(df_processed.columns)
print(df_processed.count())

df_dups=df_processed.duplicated(subset=['order_id'])
print(f"no of duplicates in order_id:{df_dups.sum()}")

print(df_processed.is_low_score)

# Define thresholds used in questions
LATE_DELIVERY_THRESHOLD_STRICT = 5 # days for Q3.1, Q4.1
LATE_DELIVERY_THRESHOLD_BASIC = 0 # days for Q2.3, Q5.1
HIGH_PRICE_THRESHOLD = 200
HIGH_PAYMENT_THRESHOLD = 300
HIGH_INSTALLMENT_THRESHOLD = 3
MANY_ITEMS_THRESHOLD = 2
LONG_RESPONSE_THRESHOLD = 72 # hours

# Get total number of rows for calculations
total_rows = df_processed.shape[0]
print(f"Total rows in df_processed: {total_rows}")
if total_rows == 0:
    print("Error: DataFrame is empty. Cannot calculate probabilities.")
    exit()

print("\n--- Calculating Probabilities ---")

# =====================================================
# 1. Basic Probability: P(A)
# =====================================================
print("\n\n=== 1. Basic Probability ===")
# --- Question 1.1: P(review_score == 5) ---
count_score_5=(df_processed['review_score']==5).sum()
print(f"Probability of review score 5: p(count_score_5)/total_rows={count_score_5/total_rows}")
#using pandas directly we can do the below
print(f"P(review_score_5):{(df_processed['review_score']==5).mean()}")


# --- Question 1.2: P(payment_type == 'boleto') ---
count_boleto=(df_processed['payment_type']=='boleto').sum()
print(f"P(payment_type=='boleto'):{count_boleto/total_rows}")
#using pandas directly we can do the below
print(f"P(payment_type=='boleto'):{(df_processed['payment_type']=='boleto').mean()}")

# --- Question 1.3: P(delivery_days > 20) ---
count_delivery=(df_processed['delivery_days']>20).sum()
print(f"P(delivery_days>20):{count_delivery/total_rows}")
#using pandas directly we can do the below
print(f"P(delivery_days>20):{(df_processed['delivery_days']>20).mean()}")




# =====================================================
# 2. Joint Probability: P(A and B)
# =====================================================
print("\n\n=== 2. Joint Probability ===")

# --- Question 2.1: P(State = SP AND Low Score = 1) ---
count_sp_low=((df_processed['customer_state']=='SP') & (df_processed['is_low_score']==1)).sum()
print(f"Probability(State=SP and is_low_score=1):{count_sp_low/total_rows}")
#using pandas directly
print(f"Probability(State=SP and is_low_score=1):{((df_processed['customer_state']=='SP')&(df_processed['is_low_score']==1)).mean()}")


# --- Question 2.3: P(Late Delivery > 0 AND Total Price > 200) ---
count_late_high=((df_processed['delivery_diff_days']>LATE_DELIVERY_THRESHOLD_BASIC) & (df_processed['total_price']>HIGH_PRICE_THRESHOLD)).sum()
print(f"Probability(Late Delivery > 0 AND Total Price > 200):{count_late_high/total_rows}")
#using pandas directly
print(f"Probability(Late Delivery > 0 AND Total Price > 200):{((df_processed['delivery_diff_days']>LATE_DELIVERY_THRESHOLD_BASIC)&(df_processed['total_price']>HIGH_PRICE_THRESHOLD)).mean()}")



# =====================================================
# 3. Conditional Probability: P(A | B)
# =====================================================
print("\n\n=== 3. Conditional Probability ===")

# --- Question 3.1: P(Low Score = 1 | Late Delivery > 5 days) ---
count_low_late=((df_processed['delivery_diff_days']>5) & (df_processed['is_low_score']==1)).sum()
count_late=(df_processed['delivery_diff_days']>5).sum()

print(f"Probability(Low Score = 1 | Late Delivery > 5 days):{count_low_late/count_late}")




# =====================================================
# 4. Bayes' Theorem: P(A | B) vs P(B | A)
# =====================================================
print("\n\n=== 4. Bayes' Theorem ===")

# --- Question 4.1: P(Late Delivery > 5 days | Low Score) ---
# Formula: P(Late>5 | Low) = [P(Low | Late>5) * P(Late>5)] / P(Low)
P_low=df_processed['is_low_score'].mean()
P_late_greater_than_5=(df_processed['delivery_diff_days']>5).mean()
count_low_and_late=((df_processed['is_low_score']==1) & (df_processed['delivery_diff_days']>5)).sum()
count_late=(df_processed['delivery_diff_days']>5).sum()
P_low_given_late=count_low_and_late/count_late
print(f"Probability(Late Delivery > 5 days | Low Score):{P_low_given_late*P_late_greater_than_5/P_low}")






# =====================================================
# 5. Independence vs. Dependence
# =====================================================
tolerance=0.01

#test 1 ----------> P(A AND B)=P(A) * P(B)
p_low_score=(df_processed['is_low_score']==1).mean()
p_late_delivery=(df_processed['delivery_diff_days']>LATE_DELIVERY_THRESHOLD_BASIC).mean()
p_low_and_late_multiply=p_low_score*p_late_delivery
p_low_and_late=((df_processed['is_low_score']==1) & (df_processed['delivery_diff_days']>LATE_DELIVERY_THRESHOLD_BASIC)).mean()

if np.isclose(p_low_and_late,p_low_and_late_multiply,atol=tolerance):
    print("Test 1: Approximately Independent")
else:
    print("Test 1:Dependent")



#test 2 ----------> P(A|B)=P(A) A=low,B=late P(A/B)=P(A intersection B)/P(B)
p_low_given_late=p_low_and_late/p_late_delivery

if np.isclose(p_low_given_late,p_low_score,atol=tolerance):
    print("Test 2: Approximately Independent")
else:
    print("Test 2:Dependent")





















