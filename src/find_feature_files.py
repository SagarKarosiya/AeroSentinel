from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"

def main():
    print("Searching for feature mapping files...\n")

    for path in DATA.rglob("*.csv"):
        name = path.name.lower()

        if "feature" in name or "channel" in name or "dictionary" in name:
            print(path.relative_to(ROOT))

if __name__ == "__main__":
    main()