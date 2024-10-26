
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
from sklearn.metrics import f1_score 
from Feature_Engineering import sample


def test_data(df_kpi):
        
    data = sample(df_kpi, "Status")

    data['Lag1'] = data['KPI_Value'].shift(1)
    data['Lag2'] = data['KPI_Value'].shift(2)
    data['Lag3'] = data['KPI_Value'].shift(3)
    data['Lag4'] = data['KPI_Value'].shift(4)
    data['Lag5'] = data['KPI_Value'].shift(5)
    data = data.fillna(0)


    # Initialize the KNeighborsClassifier with specified parameters
    model = RandomForestClassifier(max_depth=29, min_samples_leaf=1, min_samples_split=12, n_estimators=252)
    split_value = int(len(data) * 0.8)


    X = data[["KPI_Value", "Lag1","Lag2", "Lag3", "Lag4", "Lag5", "Timestamp"]]
    y = data["Status"]


    X_train, X_test = X[:split_value], X[split_value:]
    y_train, y_test = y[:split_value], y[split_value:]

    # Fit the model with your data (replace X and y with your dataset)
    model.fit(X_train[["KPI_Value", "Lag1","Lag2", "Lag3", "Lag4", "Lag5"]],y_train)

    y_pred = model.predict(X_test[["KPI_Value", "Lag1","Lag2", "Lag3", "Lag4", "Lag5"]])

    f1 = f1_score(y_test, y_pred)

    print('f1 score',f1)

    with mlflow.start_run():
    # Log model parameters
        
        # Log parameters with MLflow
        mlflow.log_param("max_depth", model.max_depth)
        mlflow.log_param("min_samples_leaf", model.min_samples_leaf)
        mlflow.log_param("min_samples_split", model.min_samples_split)
        mlflow.log_param("n_estimators", model.n_estimators)
        
        # Log F1 score
        mlflow.log_metric("f1_score", f1)
        
        # Log the model
        mlflow.sklearn.log_model(model, "random_forest_model")

        print(f"Model and parameters logged with F1 score: {f1}")
            
    return f1