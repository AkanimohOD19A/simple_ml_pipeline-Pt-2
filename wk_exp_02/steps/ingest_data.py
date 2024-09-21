## Import Libraries
import os
import pandas as pd
from zenml import step
from typing import Annotated

## Data Dependencies
df_pth = 'https://github.com/ybifoundation/Dataset/raw/main/Salary%20Data.csv'
local_pth = "./datasets/SalaryData.csv"
# local_model_register = "./model_performance.json"
# local_model_pth = "./models"


## Ingestion Step
@step(enable_cache=True)
def load_data() -> Annotated[pd.DataFrame, "Salary_data"]:
    if os.path.exists(local_pth):
        df = pd.read_csv(local_pth)
    else:
        os.makedirs("./datasets", exist_ok=True)
        df = pd.read_csv(df_pth)
        df.to_csv("./datasets/SalaryData.csv", index=False)

    return df
