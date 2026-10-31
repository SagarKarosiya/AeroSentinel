
from data_loader import load_flight, load_split


def main():
    print("=" * 60)
    print("AEROSENTINEL TELEMETRY LOADER TEST")
    print("=" * 60)

    for hardware in ("holybro", "speedybee"):

        print(f"\n{'-' * 50}")
        print(hardware.upper())

        train_flights = load_split(hardware, "train")

        print("Training flights:", len(train_flights))

        flight_id = train_flights.iloc[0]["flight_id"]

        df, metadata = load_flight(hardware, flight_id)

        print("Test flight:", flight_id)
        print("Split:", metadata["split"])
        print("Rows:", len(df))
        print("Columns:", len(df.columns))

        print("\nFirst 5 timestamps:")
        print(df["timestamp_s"].head().tolist())

        print("\nMissing values (top 10 columns):")
        print(
            df.isna()
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .to_string()
        )

        print("\nData types:")
        print(df.dtypes.value_counts().to_string())

    print("\nLoader test completed.")


if __name__ == "__main__":
    main()
