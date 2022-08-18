# Databricks notebook source
# MAGIC %md
# MAGIC ## Initial Data Table Setup
# MAGIC 
# MAGIC Initial Setup done in main notebook; here we simply change to the right database.

# COMMAND ----------

# MAGIC %sql
# MAGIC USE ggw_churndb;

# COMMAND ----------

# MAGIC %md ## Basic Data Standardization  
# MAGIC   
# MAGIC Change types and move from categorical strings to numbers or booleans

# COMMAND ----------

import pyspark.sql.functions as F

# Read Source file
telco_df = (spark.read 
                 .format("csv")
                 .option("header", True)
                 .option("inferSchema", True)
                 .load("/tmp/ML/telco_churn/Telco-Customer-Churn.csv")
           )

# Standardize datatypes (e.g. numbers or strings to booleans, etc.)

# 0/1 -> boolean
telco_df = telco_df.withColumn("SeniorCitizen", F.col("SeniorCitizen") == 1)

# Yes/No -> boolean
for yes_no_col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
  telco_df = telco_df.withColumn(yes_no_col, F.col(yes_no_col) == "Yes")

# Churn -> 0/1 
telco_df = telco_df.withColumn("Churn", F.when(F.col("Churn") == "Yes", 1).otherwise(0))

# Contract categorical -> duration in months
telco_df = (telco_df.withColumn("Contract",
                      F.when(F.col("Contract") == "Month-to-month", 1)
                       .when(F.col("Contract") == "One year", 12)
                       .when(F.col("Contract") == "Two year", 24))
           )

# Empty TotalCharges -> NaN
telco_df = (telco_df.withColumn("TotalCharges",
                      F.when(F.length(F.trim(F.col("TotalCharges"))) == 0, None)
                       .otherwise(F.col("TotalCharges").cast('double')))
           )


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
  return (data.select(service_cols)
              .fillna({"TotalCharges": 0.0})
              .withColumn("NumOptionalServices",num_optional_services("OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"))
              .withColumn("AvgPriceIncrease",F.when(F.col("tenure") > 0, (F.col("MonthlyCharges") - (F.col("TotalCharges") / F.col("tenure"))))
                                              .otherwise(0.0))
         )

# COMMAND ----------

# Compute Service Dataframe based on above
service_df = compute_service_features(telco_df)

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()

service_features_table = fs.create_table(
  name='ggw_churndb.service_features',
  primary_keys='customerID',
  schema=service_df.schema,
  description='GGW Telco customer services')

# COMMAND ----------

# MAGIC %md
# MAGIC Note: If you need to re-create the `service_features` table for any reason, the feature table has to be deleted manually from the Feature Store tab UI before `create_table` can run again.

# COMMAND ----------

fs.write_table("ggw_churndb.service_features", service_df, "overwrite")
