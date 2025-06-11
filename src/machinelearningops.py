import numpy as np
import pandas as pd
from collections import Counter
import math
import random

def accuracy_score(y_test,y_pred):
    correct=sum(y_test==y_pred)
    return correct/(len(y_test) if len(y_test)>0 else 0)

def mean_squared_error(y_test,y_pred):
    return np.mean((y_test-y_pred)**2)

def mean_absolute_error(y_test,y_pred):
    return np.mean(np.abs(y_test-y_pred))

def rmse(y_test,y_pred):
    return np.sqrt(mean_squared_error(y_test,y_pred))

def r2_score(y_test,y_pred):
    numerator=np.sum((y_test-y_pred)**2)
    denominator=np.sum((y_test-np.mean(y_test))**2)
    return 1-(numerator/denominator)

def classification_report(y_test,y_pred):
    tp=sum((y_test==1) & (y_pred==1))
    tn=sum((y_test==0) & (y_pred==0))
    fp=sum((y_test==0) & (y_pred==1))
    fn=sum((y_test==1) & (y_pred==0))
    accuracy=(tp+tn)/(tp+tn+fp+fn)
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    f1_score=2*(precision*recall)/(precision+recall)
    return accuracy,precision,recall,f1_score

def confusion_matrix(y_test,y_pred):
    TP=TN=FP=FN=0
    for actual,predicted in zip(y_test,y_pred):
        if actual==1 and predicted==1:
            TP+=1
        elif actual==0 and predicted==0:
            TN+=1
        elif actual ==0 and predicted==1:
            FP+=1
        else:
            FN+=1
    return [
        [TP,FP],
        [FN,TN]
    ]


class LinearRegression:
    def __init__(self,learning_rate,n_iterations):
        self.learning_rate=learning_rate
        self.n_iterations=n_iterations
        self.weights=None
        self.bias=None
        

    def fit(self,X,y):
        n_samples,n_features=X.shape
        self.weights=np.zeros(n_features)
        self.bias=0
        self.cost_history=[]

        for _ in range(self.n_iterations):

            y_pred=np.dot(X,self.weights)+self.bias

            cost=(1/(2*n_samples))*np.sum((y_pred-y)**2)
            self.cost_history.append(cost)

            dw=(1/n_samples)*np.dot(X.T,(y_pred-y))
            db=(1/n_samples)*np.sum(y_pred-y)

            self.weights-=self.learning_rate*dw
            self.bias-=self.learning_rate*db


    def predict(self,X):
        if self.weights is None or self.bias is None:
            raise Exception("Model hasnot been trained yet")
        return np.dot(X,self.weights)+self.bias
    

class LogisticRegression:
    def __init__(self,learning_rate,n_iterations):
        self.learning_rate=learning_rate
        self.n_iterations=n_iterations
        self.weights=None
        self.bias=None
        self.cost_history=[]

    def _sigmoid(self,z):
        z_clipped=np.clip(z,-500,500)
        return 1/(1+np.exp(-z_clipped))
    def fit(self,X,y):
        n_samples,n_features=X.shape
        self.weights=np.zeros(n_features)
        self.bias=0

        for _ in range(self.n_iterations):

            linear_output=np.dot(X,self.weights)+self.bias
            y_pred=self._sigmoid(linear_output)
            epsilon = 1e-9 
            cost=-(1/n_samples)*np.sum(y*np.log(y_pred+epsilon)+(1-y)*np.log(1-y_pred+epsilon))
            self.cost_history.append(cost)

            dw=(1/n_samples)*np.dot(X.T,(y_pred-y))
            db=(1/n_samples)*np.sum(y_pred-y)

            self.weights-=self.learning_rate*dw
            self.bias-=self.learning_rate*db

    def predict_proba(self,X):
        if self.weights is None or self.bias is None:
            raise Exception("Model has not been trained yet")
        linear_output=np.dot(X,self.weights)+self.bias
        y_prediction=self._sigmoid(linear_output)
        return np.vstack((1-y_prediction,y_prediction)).T
    
    def predict(self,X):
        probabilities=self.predict_proba(X)[:,1]
        return (probabilities>=0.5).astype(int)

   
# K-Nearest Neighbors (KNN) is a supervised machine learning algorithm used for classification and regression
class KNNClassifier:
    def __init__(self,k=3):
        self.k=k
        self.X_train=None
        self.y_train=None

    def fit(self,X,y):
        self.X_train=X
        self.y_train=y
    
    def euclidean(self,x1,x2):
        return np.sqrt(np.sum((x1-x2)**2))
    
    def _predict(self,x_test):
        distances=[self.euclidean(x_test,x_train) for x_train in self.X_train]
        k_indices=np.argsort(distances)[:self.k]

        k_labels=[self.y_train[i] for i in k_indices]
        k_most=Counter(k_labels).most_common(1)
        return k_most[0][0]
    
    def predict(self,X_test):
        predictions=[self._predict(x_test) for x_test in X_test]
        return np.array(predictions)


class Node:
    def __init__(self,feature=None,threshold=None,value=None,left=None,right=None):
        self.feature=feature
        self.threshold=threshold
        self.value=value
        self.left=left
        self.right=right

    def is_leaf_node(self):
        return self.value is not None

class DecisionTreeClassifier:
    def  __init__(self,min_samples=2,max_depth=2,n_features=None):
        self.min_samples=min_samples
        self.max_depth=max_depth
        self.n_features=n_features
        self.root=None

    def gini_impurity(self,y):
        hist=np.bincount(y)
        ps=hist/len(y)
        return 1-np.sum([p**2 for p in ps if p>0])
    
    def _find_best_split(self,X,y):
        n_samples,n_features_data=X.shape
        best_gain=-1
        split_feat,split_thresh=None,None

        for feat_idx in range(n_features_data):
            X_column=X[:,feat_idx]
            thresholds=np.unique(X_column)

            for threshold in thresholds:
                parent_impurity=self.gini_impurity(y)

                left_indices=np.argwhere(X_column<=threshold).flatten()
                right_indices=np.argwhere(X_column>threshold).flatten()

                n_left,n_right=len(left_indices),len(right_indices)
                if n_left==0 or n_right==0:
                    continue
                

                child_impurity=(n_left/n_samples)*self.gini_impurity(y[left_indices])+(n_right/n_samples)*self.gini_impurity(y[right_indices])
                information_gain=parent_impurity-child_impurity

                if information_gain>best_gain:
                    best_gain=information_gain
                    split_feat,split_thresh=feat_idx,threshold

        return split_feat,split_thresh



    def build_tree_recursively(self,X,y,depth):
        n_samples,n_features_data=X.shape
        n_labels=len(np.unique(y))

        if (depth>=self.max_depth or n_labels==1 or n_samples<=self.min_samples):
            leaf_value=self._most_common_label(y)
            return Node(value=leaf_value)
        best_feat,best_thresh=self._find_best_split(X,y)

        if best_feat is None:
            leaf_value=self._most_common_label(y)
            return Node(value=leaf_value)
        
        left_indices=np.argwhere(X[:,best_feat]<=best_thresh).flatten()
        right_indices=np.argwhere(X[:,best_feat]>best_thresh).flatten()

        left_subtree=self.build_tree_recursively(X[left_indices,:],y[left_indices],depth+1)
        right_subtree=self.build_tree_recursively(X[right_indices,:],y[right_indices],depth+1)

        return Node(best_feat,best_thresh,value=None,left=left_subtree,right=right_subtree)
    
    def _most_common_label(self,y):
        counter=Counter(y)
        most_common=counter.most_common(1)
        return most_common[0][0]
    
    def fit(self,X,y):

        y_int=y.astype(int) if pd.api.types.is_numeric_dtype(y) else y
        self.root=self.build_tree_recursively(X,y_int,0)

    def predict(self,X):

        return np.array([self.traverse_tree(self.root,x) for x in X])

    def traverse_tree(self,node,x):
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature]<=node.threshold:
            return self.traverse_tree(node.left,x)
        else:
            return self.traverse_tree(node.right,x)
        


    

# A Random Forest is an ensemble learning method that builds multiple decision trees and aggregates their results to improve accuracy and prevent overfitting.
# n_estimators: Number of decision trees to train in the forest.
# min_samples_split: Minimum samples required to split a node (passed to individual trees).
# max_depth: Maximum depth of each tree.

class RandomForestClassifier:
    def __init__(self,n_estimators=100,min_samples=2,max_depth=100,n_features=None):
        self.n_estimators=n_estimators
        self.min_samples=min_samples
        self.max_depth=max_depth
        self.n_features=n_features
        self.trees=[]


    def bootstrap_sample(self,X,y):
        n_samples=X.shape[0]
        idxs=np.random.choice(n_samples,size=n_samples,replace=True)
        return X[idxs],y[idxs]
    
    def fit(self,X,y):
        self.trees=[]

        for _ in range(self.n_estimators):

            X_sample,y_sample=self.bootstrap_sample(X,y)
            tree=DecisionTreeClassifier(min_samples=self.min_samples,max_depth=self.max_depth,n_features=self.n_features)
            tree.fit(X_sample,y_sample)
            self.trees.append(tree)
        
    def predict(self,X):
        tree_predictions=np.array([tree.predict(X)for tree in self.trees])
        #tree_predictions has shape (n_estimators,n_samples_Xtest)
        #now we take its transpose to find the most common label for each sample
        tree_label=[Counter(tree_pred).most_common(1)[0][0] for tree_pred in tree_predictions.T]
        return np.array(tree_label)
    


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer

    try:
        df = pd.read_csv('/home/teena/Documents/all_in_one/Datascience/olist_abt_processed_output.csv')
        df = df.sample(n=10000, random_state=42)
        print(f"\n--- Starting Example Usage with data shape: {df.shape} ---")
    except FileNotFoundError:
        print("ERROR: Data file not found.")
        exit()

    # Data preparation
    categorical_features = ['customer_state', 'payment_type']
    numerical_features = [
        'delivery_days', 'delivery_diff_days', 'approval_hours',
        'processing_hours', 'carrier_hours', 'review_response_hours',
        'payment_installments', 'payment_value', 'payment_count', 'total_price',
        'total_freight', 'num_items', 'distinct_sellers',
        'avg_product_weight_g', 'avg_product_volume_cm3'
    ]
    
    ids_to_drop = ['order_id', 'review_id', 'customer_id', 'customer_unique_id']
    timestamps_to_drop = [col for col in df.columns if 'timestamp' in col or 'date' in col]
    other_cols_to_drop = ['review_score', 'customer_city', 'customer_zip_code_prefix', 'product_category_name_english']

    df_model = df.drop(columns=ids_to_drop + timestamps_to_drop + other_cols_to_drop, errors='ignore')
    df_model.dropna(subset=['is_low_score'] + categorical_features + numerical_features, inplace=True)

    # Regression Example
    print("\n--- Linear Regression Example ---")
    target_regr = 'review_score'
    if target_regr in df.columns:
        X_regr = df_model[numerical_features].values
        y_regr = df[target_regr].loc[df_model.index].values
        
        X_train_regr, X_test_regr, y_train_regr, y_test_regr = train_test_split(
            X_regr, y_regr, test_size=0.2, random_state=42
        )
        
        scaler_regr = StandardScaler()
        X_train_regr_scaled = scaler_regr.fit_transform(X_train_regr)
        X_test_regr_scaled = scaler_regr.transform(X_test_regr)

        lin_reg = LinearRegression(learning_rate=0.01, n_iterations=1000)
        lin_reg.fit(X_train_regr_scaled, y_train_regr)
        y_pred_regr = lin_reg.predict(X_test_regr_scaled)

        print(f"MSE: {mean_squared_error(y_test_regr, y_pred_regr):.4f}")
        print(f"R²: {r2_score(y_test_regr, y_pred_regr):.4f}")

    # Classification Examples
    print("\n--- Classification Models ---")
    target_clas = 'is_low_score'
    if target_clas in df_model.columns:
        X = df_model[categorical_features + numerical_features]
        y = df_model[target_clas].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])
        
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        def print_report(model_name, y_true, y_pred):
            acc = accuracy_score(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred)
            accuracy, precision, recall, f1 = classification_report(y_true, y_pred)
            
            print(f"\n{model_name} Results:")
            print(f"Accuracy: {acc:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print("Confusion Matrix:")
            print(f"[[TP: {cm[0][0]}  FP: {cm[0][1]}]")
            print(f" [FN: {cm[1][0]}  TN: {cm[1][1]}]]")

        # Logistic Regression
        log_reg = LogisticRegression(learning_rate=0.1, n_iterations=1000)
        log_reg.fit(X_train_processed, y_train)
        print_report("Logistic Regression", y_test, log_reg.predict(X_test_processed))

        # KNN
        knn = KNNClassifier(k=5)
        knn.fit(X_train_processed, y_train)
        print_report("KNN", y_test, knn.predict(X_test_processed))

        # Decision Tree
        dt = DecisionTreeClassifier(max_depth=10, min_samples=10)
        dt.fit(X_train_processed, y_train)
        print_report("Decision Tree", y_test, dt.predict(X_test_processed))

        # Random Forest
        rf = RandomForestClassifier(n_estimators=10, max_depth=8, min_samples=15)
        rf.fit(X_train_processed, y_train)
        print_report("Random Forest", y_test, rf.predict(X_test_processed))