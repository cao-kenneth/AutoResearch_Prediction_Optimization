import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def resolution_to_binary(x):
    x = str(x).strip().lower()
    if x == "yes":
        return 1
    elif x == "no":
        return 0
    else:
        raise ValueError(f"Unexpected resolution: {x}")


def prob_str_to_decimal(x):
    x = str(x).strip().replace("%", "")
    return float(x) / 100.0


def format_forecast_date(date_value):
    return pd.to_datetime(date_value).strftime("%B %d, %Y")


def main():
    df = pd.read_csv("RawData.csv")

    # deterministic 60/40 split, stratified by Resolution
    train_df, test_df = train_test_split(
        df,
        test_size=0.40,
        random_state=RANDOM_STATE,
        stratify=df["Resolution"]
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # keep only the formatted train dataset for baseline use
    clean_train = train_df.copy()

    clean_train["forecast_date_formatted"] = clean_train["Forecast_Date"].apply(format_forecast_date)
    clean_train["resolution_binary"] = clean_train["Resolution"].apply(resolution_to_binary)
    clean_train["community_prob"] = clean_train["ForecastDate_Probability"].apply(prob_str_to_decimal)

    clean_train.to_csv("train.csv", index=False)

    test_df.to_csv("test_40_unused.csv", index=False)

    print(f"Total rows: {len(df)}")
    print(f"Train rows (60%): {len(clean_train)}")
    print(f"Test rows (40%): {len(test_df)}")
    print("Saved cleaned training data to train.csv")
    print("Saved unused test data to test_40_unused.csv")


if __name__ == "__main__":
    main()