## This is my first end to end project


!set MLFLOW_TRACKING_URI=https://dagshub.com/LokeshKumarDas/ML_Project.mlflow
!set MLFLOW_TRACKING_USERNAME=LokeshKumarDas


import dagshub
dagshub.init(repo_owner='LokeshKumarDas', repo_name='ML_Project', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)