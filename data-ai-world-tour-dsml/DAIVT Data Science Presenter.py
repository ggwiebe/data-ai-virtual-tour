# Databricks notebook source
# MAGIC %md 
# MAGIC 
# MAGIC # Welcome!  
# MAGIC <br>
# MAGIC <img src="https://github.com/PawaritL/data-ai-world-tour-dsml-jan-2022/blob/main/dsml-header.png?raw=true" width="63.5%">

# COMMAND ----------

# MAGIC %md
# MAGIC # Friendly Neighbourhood Tips  
# MAGIC 
# MAGIC To run a cell: press **Shift + Enter**  
# MAGIC 
# MAGIC To trigger autocomplete: press **Tab** (after entering a completable object)  
# MAGIC <img src="https://docs.databricks.com/_images/notebook-autocomplete-object.png" width="10%">  
# MAGIC 
# MAGIC To display Python docstring hints: press **Shift + Tab** (after entering a completable object)   
# MAGIC <img src="https://docs.databricks.com/_images/python-docstring.png" width="30%">

# COMMAND ----------

# MAGIC %md
# MAGIC # Setup
# MAGIC 
# MAGIC **Please run the cells below via: Shift + Enter**  
# MAGIC Stop when you reach the **Exploration** section
# MAGIC 
# MAGIC This notebook contains setup code that would have been run outside of the core data science flow. These are details that aren't part of the data science demo. Execute these first, to establish some tables and resources that will be used in the demo below.
# MAGIC 
# MAGIC **Be sure that you are running Databricks ML Runtime 10.2 or greater**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initial Data Table Setup
# MAGIC 
# MAGIC This sets up the `demographic` table, which is the initial data set considered by the data scientist. It would have been created by data engineers, in the narrative. The data set is available at https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE DATABASE IF NOT EXISTS churndb;
# MAGIC DROP TABLE IF EXISTS churndb.demographic;
# MAGIC DROP TABLE IF EXISTS churndb.service_features;
# MAGIC DROP TABLE IF EXISTS churndb.demographic_service

# COMMAND ----------

# MAGIC %sh mkdir -p /dbfs/tmp/ML/telco_churn/; wget -O /dbfs/tmp/ML/telco_churn/Telco-Customer-Churn.csv https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

# COMMAND ----------

import pyspark.sql.functions as F

telco_df = spark.read.option("header", True).option("inferSchema", True).csv("/tmp/ML/telco_churn/Telco-Customer-Churn.csv")

# 0/1 -> boolean
telco_df = telco_df.withColumn("SeniorCitizen", F.col("SeniorCitizen") == 1)
# Yes/No -> boolean
for yes_no_col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
  telco_df = telco_df.withColumn(yes_no_col, F.col(yes_no_col) == "Yes")
telco_df = telco_df.withColumn("Churn", F.when(F.col("Churn") == "Yes", 1).otherwise(0))

# Contract categorical -> duration in months
telco_df = telco_df.withColumn("Contract",\
    F.when(F.col("Contract") == "Month-to-month", 1).\
    when(F.col("Contract") == "One year", 12).\
    when(F.col("Contract") == "Two year", 24))
# Empty TotalCharges -> NaN
telco_df = telco_df.withColumn("TotalCharges",\
    F.when(F.length(F.trim(F.col("TotalCharges"))) == 0, None).\
    otherwise(F.col("TotalCharges").cast('double')))

telco_df.select("customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "Churn").write.format("delta").saveAsTable("churndb.demographic")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Service Feature Table Setup
# MAGIC 
# MAGIC This sets up the feature store table `service_features`, which again is presumed to have been created earlier by data engineers or other teams.

# COMMAND ----------

def compute_service_features(data):
  # Count number of optional services enabled, like streaming TV
  @F.pandas_udf('int')
  def num_optional_services(*cols):
    return sum(map(lambda s: (s == "Yes").astype('int'), cols))
  
  # Below also add AvgPriceIncrease: current monthly charges compared to historical average
  service_cols = [c for c in data.columns if c not in ["gender", "SeniorCitizen", "Partner", "Dependents", "Churn"]]
  return data.select(service_cols).fillna({"TotalCharges": 0.0}).\
    withColumn("NumOptionalServices",
        num_optional_services("OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies")).\
    withColumn("AvgPriceIncrease",
        F.when(F.col("tenure") > 0, (F.col("MonthlyCharges") - (F.col("TotalCharges") / F.col("tenure")))).otherwise(0.0))

service_df = compute_service_features(telco_df)

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()

service_features_table = fs.create_table(
  name='churndb.service_features',
  primary_keys='customerID',
  schema=service_df.schema,
  description='Telco customer services')

# COMMAND ----------

# MAGIC %md
# MAGIC Note: If you need to re-create the `service_features` table for any reason, the feature table has to be deleted manually from the Feature Store tab UI before `create_table` can run again.

# COMMAND ----------

fs.write_table("churndb.service_features", service_df)

# COMMAND ----------

# MAGIC %md
# MAGIC At this point you should have:
# MAGIC 
# MAGIC - a database table at `churndb.demographic`, visible in the Data tab, which contains customer demographic data (and churn status)
# MAGIC - a feature table in the Feature Store tab called `churndb.service_features` with customer service-related info -- try it!
# MAGIC 
# MAGIC Last, but not least, we'll need the latest `scikit-learn`:

# COMMAND ----------

# MAGIC %pip install scikit-learn==1.0.2

# COMMAND ----------

# MAGIC %md
# MAGIC The next cell just stops execution in case you clicked "Run All"!

# COMMAND ----------

dbutils.notebook.exit("stop")

# COMMAND ----------

# MAGIC %md
# MAGIC # Exploration
# MAGIC 
# MAGIC Welcome to Databricks! This session will illustrate a fictional, simple, but representative day in the life of a data scientist on Databricks, who starts with data and ends up with a basic production service.
# MAGIC 
# MAGIC ## Problem: Churn  
# MAGIC <img src="https://databricks.com/wp-content/uploads/2020/08/blog-profit-drive-retention-1-min.png" width="38%">
# MAGIC 
# MAGIC Imagine the case of a startup telecom company, with customers who unfortunately sometimes choose to terminate their service. It would be useful to predict when a customer might churn, to intervene. Fortunately, the company has been diligent about collecting data about customers, which might be predictive. This is new territory, the first time the company has tackled the problem. Where to start?
# MAGIC 
# MAGIC ## Data Exploration
# MAGIC 
# MAGIC We can start by simply reading the data and exploring it. There's already some useful information in the `demographic` table: customer ID, whether they have churned (or not, yet), and basic demographic information:

# COMMAND ----------

display(spark.read.table("churndb.demographic"))

# COMMAND ----------

# MAGIC %md
# MAGIC Do the normal, predictable things. Compute summary stats. Plot some values. See what's what.

# COMMAND ----------

display(spark.read.table("churndb.demographic").summary())

# COMMAND ----------

display(spark.read.table("churndb.demographic")) # check out other Plot Options too

# COMMAND ----------

# MAGIC %md
# MAGIC This is easy, but frankly not that informative. Looking deeper would require writing some more code to explore, and, after all we do want to get on to trying out modeling too. Rather than continue, how about a coffee break while the machines do some work?
# MAGIC <img src="https://github.com/PawaritL/data-ai-world-tour-dsml-jan-2022/blob/main/automl-motivation.png?raw=true" width="69%">  
# MAGIC 
# MAGIC ### Try it out!
# MAGIC Use AutoML to explore this data set and build a baseline model before continuing:  
# MAGIC <img src="https://databricks.com/wp-content/uploads/2021/05/automl-blog-img-7.jpg" width="50%">  
# MAGIC <br>
# MAGIC - From the Machine Learning home screen, click "Start AutoML" at the top center
# MAGIC - Choose your cluster for "Compute"
# MAGIC - Choose "Classification" type
# MAGIC - Browse to database table `churndb.demographic`
# MAGIC - Under Advanced Options, **limit to 5 trial runs and 5 minutes**
# MAGIC - Run!
# MAGIC - After the 5 runs complete you can stop the experiment
# MAGIC 
# MAGIC You should see the results, logged with MLflow, in the UI.  

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## How did the baseline model perform?
# MAGIC 
# MAGIC The model was OK, but, could probably be better with more data.  
# MAGIC ***Wouldn't it be nice*** if we could enrich our models with other relevant information?  
# MAGIC [Reference: Databricks Telco Solutions Accelerator](https://databricks.com/blog/2021/02/24/solution-accelerator-telco-customer-churn-predictor.html)  
# MAGIC <img src="https://databricks.com/wp-content/uploads/2021/02/telco-accel-blog-2-new-1024x538.png" width="50%">  
# MAGIC **Question is...how can I discover these other datasets?**

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Exploration with the Feature Store
# MAGIC 
# MAGIC There is, fortunately, more data about customers available -- additional information about the services they use, as well as some other derived, aggregated data.  
# MAGIC This **"services"** dataset was previously used for another customer-related modeling task.  
# MAGIC The data is therefore available in the **Feature Store** (UI example below).  
# MAGIC <br>
# MAGIC <img src="https://databricks.com/wp-content/uploads/2021/05/fs-blog-img-2.png" width="50%">  
# MAGIC <br>
# MAGIC Why not reuse these features and build on top of this great work?  
# MAGIC Programmatically access the Feature Store to read and join everything in the **`service_features`** feature table.

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient, FeatureLookup

fs = FeatureStoreClient()

training_set = fs.create_training_set(spark.read.table("churndb.demographic"),
                                      [FeatureLookup(table_name = "churndb.service_features", lookup_key="customerID")],
                                      label=None, exclude_columns="customerID")

display(training_set.load_df())

# COMMAND ----------

# MAGIC %md
# MAGIC Save the augmented data set as a table, for AutoML.

# COMMAND ----------

training_set.load_df().write.format("delta").saveAsTable("churndb.demographic_service")

# COMMAND ----------

# MAGIC %md
# MAGIC Try the above again, this time using the augmented data set in `churndb.demographic_service`.  
# MAGIC This time, let's take a look at the **Data Exploration notebook**, and the **Best Model notebook** before proceeding.  
# MAGIC <br>
# MAGIC <img src="https://databricks.com/wp-content/uploads/2021/05/glass-box-approach-to-automl.svg" width="50%">

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Insights with SHAP  
# MAGIC 
# MAGIC Credits: [github.com/slundberg/shap](https://github.com/slundberg/shap)  
# MAGIC <img src="https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/shap_header.svg" width="50%"> 
# MAGIC 
# MAGIC As part of the exploration process, having a baseline model early helps explore the _data_ in turn.  
# MAGIC For example, the basic SHAP plots created by auto ML can be expanded to explore more of the data:

# COMMAND ----------

import mlflow
import mlflow.sklearn
from shap import KernelExplainer, summary_plot
from sklearn.model_selection import train_test_split
import pandas as pd
import shap

mlflow.autolog(disable=True)

# be sure to change the following run URI to match the best model generated by AutoML
model_uri = 'runs:/69755aad201e4d48abd021e5d6106126/model'
#model_uri = "runs:/[your run ID here!]/model"

sample = spark.read.table("churndb.demographic_service").sample(0.05, seed=42).toPandas()
data = sample.drop(["Churn"], axis=1)
labels = sample["Churn"]
X_background, X_example, _, y_example = train_test_split(data, labels, train_size=0.25, random_state=42, stratify=labels)

model = mlflow.sklearn.load_model(model_uri)

predict = lambda x: model.predict_proba(pd.DataFrame(x, columns=X_background.columns))[:,-1]
explainer = KernelExplainer(predict, X_background)
shap_values = explainer.shap_values(X=X_example, nsamples=100)

# COMMAND ----------

summary_plot(shap_values, features=X_example)

# COMMAND ----------

# MAGIC %md
# MAGIC ... or explore how churn factors differ by gender:

# COMMAND ----------

from shap import group_difference_plot

group_difference_plot(shap_values[y_example == 1], X_example[y_example == 1]['gender'] == 'Male', feature_names=X_example.columns)

# COMMAND ----------

# MAGIC %md
# MAGIC ... or 'cluster' customers by their model explanations and see whether patterns emerge. There are clearly a group of churners that tend to be on 1-month contracts, short tenure, and equally a cluster of non-churners who are on 2-year contracts and have been long time customers. But we knew that!

# COMMAND ----------

import seaborn as sns
from sklearn.manifold import TSNE

embedded = TSNE(n_components=2, learning_rate='auto', init='pca', random_state=42).fit_transform(shap_values)

sns.set(rc = {'figure.figsize':(16,9)})
sns.scatterplot(x=embedded[:,0], y=embedded[:,1], \
                style=X_example['Contract'], \
                hue=X_example['tenure'], \
                size=(y_example == 1), size_order=[True, False], sizes=(100,200))

# COMMAND ----------

dbutils.notebook.exit("stop")

# COMMAND ----------

# MAGIC %md
# MAGIC # Modeling
# MAGIC 
# MAGIC <img src="https://databricks.com/wp-content/uploads/2021/05/Glass-Box-Approach-to-AutoML-1-light.png" width="50%">
# MAGIC 
# MAGIC Auto ML generated a baseline model for us, but, we could already see it was too simplistic.  
# MAGIC From that working modeling code, the data scientist could iterate and improve it by hand.
# MAGIC 
# MAGIC **... time passes ...**
# MAGIC 
# MAGIC A few days later, we've got a condensed and improved variation on the modeling code generated by AutoML.
# MAGIC 
# MAGIC Get the latest features from the feature store:

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient, FeatureLookup

fs = FeatureStoreClient()

training_set = fs.create_training_set(spark.read.table("churndb.demographic"),
                                      [FeatureLookup(table_name="churndb.service_features", lookup_key="customerID")],
                                      label="Churn", exclude_columns="customerID")
df_loaded = training_set.load_df().toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC This is the same as the code produced by Auto ML, to define the model:

# COMMAND ----------

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

def build_model(params):
    transformers = []

    bool_pipeline = Pipeline(steps=[
            ("cast_type", FunctionTransformer(lambda df: df.astype(object))),
            ("imputer", SimpleImputer(missing_values=None, strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    transformers.append(("boolean", bool_pipeline,
                         ["Dependents", "PaperlessBilling", "Partner", "PhoneService",
                          "SeniorCitizen"]))

    numerical_pipeline = Pipeline(steps=[
            ("converter", FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors="coerce"))),
            ("imputer", SimpleImputer(strategy="mean"))
    ])
    transformers.append(("numerical", numerical_pipeline,
                         ["AvgPriceIncrease", "Contract", "MonthlyCharges", "NumOptionalServices",
                          "TotalCharges", "tenure"]))

    one_hot_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(missing_values=None, strategy="constant", fill_value="")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    transformers.append(("onehot", one_hot_pipeline,
                         ["DeviceProtection", "InternetService", "MultipleLines", "OnlineBackup", \
                          "OnlineSecurity", "PaymentMethod", "StreamingMovies", "StreamingTV",
                          "TechSupport", "gender"]))

    xgbc_classifier = XGBClassifier(
            n_estimators=int(params['n_estimators']),
            learning_rate=params['learning_rate'],
            max_depth=int(params['max_depth']),
            min_child_weight=params['min_child_weight'],
            random_state=810302555
    )

    return Pipeline([
            ("preprocessor",
             ColumnTransformer(transformers, remainder="passthrough", sparse_threshold=0)),
            ("standardizer", StandardScaler()),
            ("classifier", xgbc_classifier),
    ])


# COMMAND ----------

from sklearn.model_selection import train_test_split

target_col = "Churn"
split_X = df_loaded.drop([target_col], axis=1)
split_y = df_loaded[target_col]

X_train, X_val, y_train, y_val = train_test_split(split_X, split_y, train_size=0.9,
                                                  random_state=810302555, stratify=split_y)

# COMMAND ----------

# MAGIC %md
# MAGIC Here, we use `hyperopt` to perform some *'Auto ML'* every time the model is rebuilt, to fine-tune it. This is similar to what auto ML did to arrive at the initial baseline model.

# COMMAND ----------

from hyperopt import fmin, hp, tpe, SparkTrials, STATUS_OK
import numpy as np
from sklearn.metrics import log_loss, accuracy_score
import mlflow

def train_model(params):
  model = build_model(params)
  model.fit(X_train, y_train)
  loss = log_loss(y_val, model.predict_proba(X_val))
  mlflow.log_metrics({'log_loss': loss, 'accuracy': accuracy_score(y_val, model.predict(X_val))})
  return {'status': STATUS_OK, 'loss': loss}

search_space = {
  'max_depth'       : hp.quniform('max_depth', 3, 10, 1),
  'learning_rate'   : hp.loguniform('learning_rate', -5, -1),
  'min_child_weight': hp.loguniform('min_child_weight', 0, 2),
  'n_estimators'    : hp.quniform('n_estimators', 50, 500, 10)
}

best_params = fmin(fn=train_model, space=search_space, algo=tpe.suggest, \
                   max_evals=8, trials=SparkTrials(parallelism=8),
                   rstate=np.random.default_rng(42))

# COMMAND ----------

# MAGIC %md
# MAGIC You should see model runs recorded by MLflow appear in the Experiments sidebar at the right. It fit several models and picked the parameters that worked best, in a brief parameter tuning run as `best_params`.
# MAGIC 
# MAGIC Now, build one last model on all the data, with the best hyperparams. The model is logged in a 'feature store aware' way, so that it can perform the joins at runtime. The model doesn't need to be fed the features manually.

# COMMAND ----------

import mlflow
from mlflow.models.signature import infer_signature

mlflow.autolog(log_input_examples=True)

with mlflow.start_run() as run:
    training_set = fs.create_training_set(spark.read.table("churndb.demographic"),
                                          [FeatureLookup(table_name="churndb.service_features",
                                                         lookup_key="customerID")],
                                          label="Churn", exclude_columns="customerID")
    df_loaded = training_set.load_df().toPandas()
    split_X = df_loaded.drop([target_col], axis=1)
    split_y = df_loaded[target_col]

    model = build_model(best_params)
    model.fit(split_X, split_y)

    fs.log_model(
            model,
            "model",
            flavor=mlflow.sklearn,
            training_set=training_set,
            registered_model_name="churn",
            input_example=split_X[:100],
            signature=infer_signature(split_X, split_y))

    best_run = run.info
    
best_run

# COMMAND ----------

# MAGIC %md
# MAGIC You can examine the final model that was fit - note that the Run has a model that contains additional information about the features needed to join to use the model!
# MAGIC 
# MAGIC The process above created a new version of the registered model `churn`. It is visible in the Models tab as a registered model.  
# MAGIC **Next Step: Transition Version 1 to Staging** (this can also be done in the UI).
# MAGIC 
# MAGIC <img src="https://databricks.com/wp-content/uploads/2020/04/databricks-adds-access-control-to-mlflow-model-registry_01.jpg" width="50%">

# COMMAND ----------

import mlflow.tracking

client = mlflow.tracking.MlflowClient()

model_version = client.get_latest_versions("churn", stages=["None"])[0]
client.transition_model_version_stage("churn", model_version.version, stage="Staging")

# COMMAND ----------

dbutils.notebook.exit("stop")

# COMMAND ----------

# MAGIC 
# MAGIC %md
# MAGIC # Automated Testing
# MAGIC 
# MAGIC This section is derived from the auto-generated batch inference notebook, from the MLflow Model Registry. It loads the latest Staging candidate model and, in addition to running inference on a data set, assesses model metrics on that result and from the training run. If successful, the model is promoted to Production. This is scheduled to run as a Job, triggered manually or on a schedule - or by a webhook set up to respond to state changes in the registry.
# MAGIC 
# MAGIC Load the model and set up the environment it defines:

# COMMAND ----------

from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
import os

local_path = ModelsArtifactRepository(f"models:/churn/staging").download_artifacts("")
# if you need to, you can load other artifacts for your testing

# COMMAND ----------

# MAGIC %md
# MAGIC Assert that the model accuracy was at least 80% at training time:

# COMMAND ----------

import mlflow.tracking

client = mlflow.tracking.MlflowClient()
latest_model_detail = client.get_latest_versions("churn", stages=['Staging'])[0]
accuracy = mlflow.get_run(latest_model_detail.run_id).data.metrics['training_accuracy_score']
print(f"Training accuracy: {accuracy}")
assert(accuracy >= 0.8)

# COMMAND ----------

# MAGIC %md
# MAGIC If successful, transition model version to Production:  
# MAGIC (then let's go see it in the Model Registry UI)

# COMMAND ----------

client.transition_model_version_stage("churn", latest_model_detail.version, stage="Production", archive_existing_versions=True)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Recap: End-to-end Machine Learning workflow  
# MAGIC <br>
# MAGIC <img src="https://databricks.com/wp-content/uploads/2020/06/blog-mlflow-model-1.png">
