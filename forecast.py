import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet

INPUT_FILE = "test_task_1.csv"
OUTPUT_FILE = "forecast_task1_jan2025.csv"
PLOTS_DIR = "forecast_plots"

os.makedirs(PLOTS_DIR, exist_ok=True)

df = pd.read_csv(INPUT_FILE, sep=None, engine="python")
df.rename(columns=lambda x: x.strip(), inplace=True)

df["date_at"] = pd.to_datetime(df["date_at"], dayfirst=True, errors="coerce")
df["cashin"] = pd.to_numeric(df["cashin"], errors="coerce").fillna(0)
df["cashout"] = pd.to_numeric(df["cashout"], errors="coerce").fillna(0)

df_2024 = df[df["date_at"].dt.year == 2024].copy()
atm_list = sorted(df_2024["atm_id"].unique())

forecast_list = []
for atm in atm_list:
    sub = df_2024[df_2024["atm_id"] == atm].groupby("date_at").agg({"cashin": "sum", "cashout": "sum"})
    sub = sub.asfreq("D").fillna(0.0).reset_index()

    for col in ["cashin", "cashout"]:
        df_prophet = sub.rename(columns={"date_at": "ds", col: "y"})

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False
        )
        model.add_country_holidays(country_name='RU')
        model.fit(df_prophet)

        # прогноз
        future = model.make_future_dataframe(periods=31)
        forecast = model.predict(future)

        forecast_jan = forecast[(forecast["ds"] >= "2025-01-01") & (forecast["ds"] <= "2025-01-31")]
        tmp = pd.DataFrame({
            "atm_id": atm,
            "date": forecast_jan["ds"],
            f"{col}_pred": np.round(forecast_jan["yhat"], 2)
        })
        forecast_list.append(tmp)

        # визуализация
        fig1 = model.plot(forecast)
        plt.title(f"{col.upper()} — Прогноз ATM {atm}")
        fig1.savefig(os.path.join(PLOTS_DIR, f"{atm}_{col}_forecast.png"))
        plt.close(fig1)


forecast_df = pd.concat(forecast_list)
forecast_df = forecast_df.pivot_table(index=["atm_id", "date"],
                                      values=["cashin_pred", "cashout_pred"],
                                      aggfunc="first").reset_index()
forecast_df.to_csv(OUTPUT_FILE, index=False)
forecast_df.to_excel('forecast_task1_jan2025.xlsx', index=False)