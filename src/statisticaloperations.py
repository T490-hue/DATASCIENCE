import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import random
# import statsmodels.stats.proportion as smp
# import statsmodels.formula.api as smf # For regression
# from statsmodels.stats.power import TTestIndPower # For power analysis
from sklearn.utils import resample # For bootstrapping (can also do manually)
sns.set_style("whitegrid")
plt.rc('figure', figsize=(10, 6))
df_processed_all=pd.read_csv('/home/teena/Documents/all_in_one/Datascience/olist_abt_processed_output.csv')
print(df_processed_all.columns)
SUBSET_SIZE = 5000  # Adjust this value to control sample size
df_processed = df_processed_all.sample(n=SUBSET_SIZE, random_state=42)  # Random subset
print(f"\nAnalyzing SUBSET of {SUBSET_SIZE} rows instead of full dataset")

# --- Define Significance Level ---
ALPHA = 0.05
print(f"Using Significance Level (alpha): {ALPHA}")

# Set plot style
sns.set_style("whitegrid")
plt.rc('figure', figsize=(10, 6))

print(f"\n--- Starting Statistical Operations using {len(df_processed)} rows ---")

# =====================================================
# 1. Descriptive Statistics
# =====================================================
print("\n\n=== 1. Descriptive Statistics ===")
# Purpose: Summarize the central tendency, dispersion, and shape of the data.

# --- Q: What is the average delivery delay? (Mean/Median) ---
print("\n--- Q: Average Delivery Delay (delivery_diff_days) ---")
col='delivery_diff_days'

if col in df_processed.columns:
    median_delay=df_processed[col].median()
    mean_delay=df_processed[col].mean()

    print(f"Median Delivery Delay: {median_delay:.2f} days")
    print(f"Mean Delivery Delay: {mean_delay:.2f} days")
else:
    print(f"Column '{col}' not found.")



# --- Q: What’s the distribution of review scores? (Mode/Value Counts) ---
col='review_score'
if col in df_processed.columns:
    mode_counts=df_processed[col].mode().tolist()
    score_counts=df_processed[col].value_counts(normalize=True).sort_index()
    print(f"Mode Review Score(s): {mode_counts}")
    print("Review Score Distribution (%):")
    print(score_counts*100)

#plot
plt.figure(figsize=(10,6))
# Uses Seaborn's countplot - correct usage for visualization.
sns.countplot(data=df_processed,x=col)
plt.title('Distribution of Review Scores')
plt.xlabel('Review Score')
plt.ylabel('Count')
plt.show() # This will raise a UserWarning if run in a non-GUI environment.

# =====================================================
# 2. Hypothesis Testing
# =====================================================
print("\n\n=== 2. Hypothesis Testing ===")
# Purpose: Make decisions about population parameters based on sample data.
# Key elements: H₀ (Null), H₁ (Alternative), α (Significance Level), p-value (Evidence against H₀).
# Decision Rule: If p-value < α, reject H₀. Otherwise, fail to reject H₀.

# --- Q: Is the average delivery time significantly different from 10 days? (One-Sample T-test) ---
print("\n--- Q: Is mean delivery_days significantly different from 10? (One-Sample T-test) ---")
col='delivery_days'
hypothesized_value=12
if col in df_processed.columns:
    sample_data=df_processed[col].dropna() #sample data is pandas series object -column of df and we can use .mean() or .stddev on the series
    n=len(sample_data)

    if n>1:
        sample_mean=sample_data.mean()
        #below is the formula for computing the standard deviation of the sample ,ddof=1 is used to find pop std dev and ddof =0 is used to fin
        #sample std dev and the formula they use is root of  E(x-mean)^2/n-1\
        sample_std=sample_data.std(ddof=1)
        t_statistic=(sample_mean-hypothesized_value)/(sample_std/np.sqrt(n))
        degrees_of_freedom = n - 1

        p_value=2*(1-stats.t.cdf(abs(t_statistic),degrees_of_freedom))

        print(f"Ho={hypothesized_value}")
        print(f"H1≠ {hypothesized_value}")
        print(f"Sample Mean: {sample_mean:.2f}")
        print(f"T-statistic: {t_statistic:.3f}")
        print(f"P-value (two-tailed): {p_value:.4f}")

        print(f"Interpretation: Assuming the true mean delivery time IS {hypothesized_value} days, the probability of observing a sample mean as extreme as {sample_mean:.2f} (or more extreme) just by random chance is {p_value:.4f}.")

        if p_value<ALPHA:
            print(f"Conclusion: Reject H₀ (p < {ALPHA}). There is statistically significant evidence that the average delivery time is different from {hypothesized_value} days.")
        else:
            print(f"Conclusion: Fail to reject H₀ (p >= {ALPHA}). There is not enough statistically significant evidence to conclude the average delivery time is different from {hypothesized_value} days.")



print("\n--- Q: Do credit card users give higher review scores? (Independent T-test) ---")
col_score='review_score'
col_group='payment_type'
group_label='credit_card'
if col_score in df_processed.columns and col_group in df_processed.columns:
    group1=df_processed[df_processed[col_group]==group_label][col_score].dropna()
    group2=df_processed[df_processed[col_group]!=group_label][col_score].dropna()
    if len(group1)>1 and len(group2)>1:
        n1=len(group1)
        n2=len(group2)
        mean1=group1.mean()
        mean2=group2.mean()
        var1=group1.var(ddof=1)
        var2=group2.var(ddof=1)


        t_statistic=(mean1-mean2)/np.sqrt(var1/n1+var2/n2)

        # Calculate degrees of freedom (Welch's formula)
        dof = ((var1 / n1 + var2 / n2) ** 2) / (((var1 / n1) ** 2) / (n1 - 1) + ((var2 / n2) ** 2) / (n2 - 1))

        p_value=2*(1-stats.t.cdf(abs(t_statistic),dof))

        print(f"mean of {group1}: {mean1:.2f} (n={n1})")
        print(f"mean of other payment types: {mean2:.2f} (n={n2})")
        print(f"t statistic: {t_statistic:.3f}")
        print(f"p value: {p_value:.4f}")

        print(f"Interpretation: Assuming the true mean scores ARE equal or credit card scores are lower, the probability of observing a difference in sample means as large as ({mean1:.2f} - {mean2:.2f} = {mean1-mean2:.2f}) or larger, purely by chance, is {p_value:.4f}.")
        if p_value<ALPHA:
            print(f"Conclusion: Reject H₀ (p < {ALPHA}). There is statistically significant evidence that customers paying by credit card give higher review scores on average.")
        else:
            print(f"Conclusion: Fail to reject H₀ (p >= {ALPHA}). There is not enough statistically significant evidence to conclude that customers paying by credit card give higher review scores on average.")



#The Chi-Square Test of Independence is a statistical test used to determine whether there is a significant relationship between two categorical variables. In this case, you're testing whether
# the payment method is independent of the customer's state.
print("\n--- Q: Is payment method independent of customer state? (Chi-Square Test) ---")
print("H₀: The payment method is independent of the customer state.")
print("H₁: The payment method is dependent on the customer state.")
col1='payment_type'
col2='customer_state'
top_n_states=5
if col1 in df_processed.columns and col2 in df_processed.columns:
    top_states=df_processed[col2].value_counts().nlargest(top_n_states).index
    df_filtered=df_processed[df_processed[col2].isin(top_states)]
    contingency_table=pd.crosstab(df_filtered[col1],df_filtered[col2])
    if contingency_table.empty:
        print("Cannot perform Chi-Square test: Contingency table is empty.")
    else:
        row_totals=contingency_table.sum(axis=1).values #.values converts the series to numpy array
        col_totals=contingency_table.sum(axis=0).values
        grand_total=contingency_table.sum().sum()

        expected=np.outer(row_totals,col_totals)/grand_total #np.outer is used to compute the outer product of two vectors and each4
        #value will be the product of row and col totals of corresponding element
        chi2_stat=np.sum(((contingency_table.values-expected)**2/expected))
        dof=(contingency_table.shape[0]-1)*(contingency_table.shape[1]-1)
        p_value=1-stats.chi2.cdf(chi2_stat,dof)

        print(f"Chi-Square Statistic: {chi2_stat:.3f}")
        print(f"Degrees of Freedom: {dof}")
        print(f"P-value: {p_value:.4f}")

        print(f"Interpretation: Assuming the payment method and customer state are independent, the probability of observing an association as strong as (or stronger than) the one in our sample data purely by chance is {p_value:.4f}.")
        if p_value<ALPHA:
            print(f"Conclusion:reject Ho .There is dependency btween payment method and customer state")
        else:
            print(f"Conclusion:Fail to reject Ho. There is no statistically significant evidence of an association between payment method and customer state")



# Confidence Interval for Average Product Price
print("\n--- Q: What is the 95% confidence interval for average product price? ---")
col='total_price'
if col in df_processed.columns:
    sample_data=df_processed[col].dropna()
    n=len(sample_data)
    if n>1:
        sample_mean=sample_data.mean()
        sample_std=sample_data.std(ddof=1)
        df=n-1
        t_critical=stats.t.ppf(1-ALPHA/2,df)
        margin_of_error=t_critical* (sample_std/np.sqrt(n))
        lower_bound=sample_mean-margin_of_error
        upper_bound=sample_mean+margin_of_error

        print(f"Sample Mean: {sample_mean:.2f}")
        print(f"Sample Std Dev: {sample_std:.2f}")
        print(f"95% Confidence Interval: ({lower_bound:.2f}, {upper_bound:.2f})")
    else:
        print("Insufficient data for confidence interval.")
else:
    print(f"Column '{col}' not found.")


print("\n--- Q: Is the mean delivery time the same across categories? (ANOVA) ---")
col='delivery_days'
group_col='product_category_name_english'

if col in df_processed_all.columns and group_col in df_processed_all.columns:
    grouped_data=[df_processed_all[df_processed_all[group_col]==category][col].dropna()   for category in df_processed_all[group_col].unique()]

    if all(len(group)>1 for group in grouped_data):
        overall_mean=df_processed_all[col].mean()
        between_group_variance = np.sum([len(group) * (group.mean() - overall_mean) ** 2 for group in grouped_data]) / (len(grouped_data) - 1)

# Within-group variance
        within_group_variance = np.sum([np.sum((group - group.mean()) ** 2) for group in grouped_data]) / (len(df_processed_all) - len(grouped_data))      
        f_statistic=between_group_variance/within_group_variance
        dof_between=len(grouped_data)-1
        dof_within=len(df_processed_all)-len(grouped_data)
        p_value=1-stats.f.cdf(f_statistic,len(grouped_data)-1,len(df_processed_all[col])-len(grouped_data))

        print(f"F-statistic: {f_statistic:.4f}")
        print(f"P-value: {p_value:.6f}")

        print(f"Interpretation: Assuming the mean delivery time is the same across all categories, the probability of observing a difference in sample means as large as (or larger than) the ones in our data purely by chance is {p_value:.4f}.")
        if p_value<ALPHA:
            print(f"Conclusion: Reject H₀ (p < {ALPHA}). There is statistically significant evidence that the mean delivery time differs across categories.")
        else:
            print(f"Conclusion: Fail to reject H₀ (p >= {ALPHA}). There is no statistically significant evidence that the mean delivery time differs across categories.")
    else:
        print("Insufficient data for ANOVA test.")
else:
    print(f"Required columns ('{col}', '{group_col}') not found.")



# =====================================================
# 6. Correlation & Covariance (Scratch Implementation)
# =====================================================
def mean_data(data_list):
    n=len(data_list)
    if n==0: return float('nan')

    return sum(data_list)/n

def variance_data(data_list,sample=True):
    n=len(data_list)
    if n<2: return float('nan')
    x_bar=mean_data(data_list)
    return sum((x-x_bar)**2 for x in data_list)/ (n-1 if sample else n)

def std_dev_data(data_list,sample=True):
    if len(data_list)<2: return float('nan')
    return np.sqrt(variance_data(data_list,sample))

def covariance_data(data_x,data_y,sample=True):
    n=len(data_x)
    if n<2 or n!=len(data_y): return float('nan')
    mean_x=mean_data(data_x)
    mean_y=mean_data(data_y)
    cov= sum([(data_x[i]-mean_x)*(data_y[i]-mean_y) for i in range(n)])/(n-1 if sample else n)
    return cov


def correlation_data(data_x,data_y,sample=True):
    n=len(data_x)
    if n<2 or n!=len(data_y): return float('nan')
    std_dev_x=std_dev_data(data_x,sample)
    std_dev_y=std_dev_data(data_y,sample)
    if std_dev_x==0 or std_dev_y==0: return float('nan')
    return covariance_data(data_x,data_y,sample)/(std_dev_x*std_dev_y)

# --- Q: What’s the correlation between delivery time and review score?
col1='delivery_days'
col2='review_score'

if col1 in df_processed_all.columns and col2 in df_processed_all.columns:
    df_processed_relation=df_processed_all[[col1,col2]].dropna()
    list1=df_processed_relation[col1].tolist()
    list2=df_processed_relation[col2].tolist()

    if len(list1)>1 and len(list2)>1:
        covar=covariance_data(list1,list2)
        corr=correlation_data(list1,list2)

        print(f"Covariance: {covar:.4f}")
        print(f"Correlation: {corr:.4f}")
    else:
        print("Insufficient data for correlation.")



#BOOTSTRAPPING AND RESAMPLING

print("\n--- Q: What is the 95% confidence interval for average delivery delay (bootstrapping)? ---")
col='delivery_diff_days'
if col in df_processed_all.columns:
    sample_data=df_processed_all[col].dropna().tolist()
    n=len(sample_data)

    num_bootstrap_samples=500
    bootstrap_means=[]
    for _ in range(num_bootstrap_samples):
        resample=[random.choice(sample_data) for _ in range(n)]
        resample_mean=np.mean(resample)
        bootstrap_means.append(resample_mean)


    #we are finding the 95% confidence interval by finding the 2.5 and 97.5 percentile of the bootstrap means and lower bound gives
    #the lower end of the interval(below which 2.5% of the bootstrap means fall) and upper bound(97.5 th percentile below which 97.5 percent of the data falls and above which 2.5 % of the data falls) gives the upper end of the interval
    lower_bound=np.percentile(bootstrap_means,100*ALPHA/2)  #2.5 percentile
    upper_bound=np.percentile(bootstrap_means,100*(1-ALPHA/2)) #97.5 percentile
    bootstrap_mean=np.mean(bootstrap_means)

    print(f"Bootstrap Mean: {bootstrap_mean:.2f}")
    print(f"95% Confidence Interval: ({lower_bound:.2f}, {upper_bound:.2f})")

    plt.hist(bootstrap_means,bins=30,color='skyblue',edgecolor='black',alpha=0.7)
    plt.axvline(lower_bound,color='red',linestyle='--',label='Lower 95% CI')
    plt.axvline(upper_bound,color='red',linestyle='--',label='Upper 95% CI')
    plt.title('Bootstrap Distribution of Mean Delivery Delay')
    plt.xlabel('Mean Delivery Delay')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('bootstrap_distribution.png')
    























