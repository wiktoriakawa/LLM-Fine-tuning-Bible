"""
Wykres: średni christian_mistral dla każdego modelu,
w zależności od liczby epok, z podziałem na:
  - typ pytania: philosophical / non-philosophical
  - zbiór danych: NT, OT, Combined, NTST, ST, Base Mistral

Uruchomienie:
  python plot_christian.py

Wymaga: plików *_christian.csv (po zakończeniu score_christian.py)
"""

import csv
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

_here = Path(__file__).parent

# Mapowanie plik → (zbiór danych, liczba epok)
FILE_META = {
    "1 epoka/nt_1ep_results_christian.csv":                          ("NT",           1),
    "1 epoka/ot_1ep_results_christian.csv":                          ("OT",           1),
    "1 epoka/results_all_1epok_christian.csv":                       ("Full",         1),
    "5 epok/nt_results_christian.csv":                               ("NT",           5),
    "5 epok/ot_results_christian.csv":                               ("OT",           5),
    "5 epok/combined_results_christian.csv":                         ("Full",         5),
    "10 epok/responses.judged_mistral(NT-10)_christian.csv":         ("NT",          10),
    "10 epok/responses.judged_mistral(NTST-10)_christian.csv":       ("Full",        10),
    "10 epok/responses.judged_mistral(ST-10)_christian.csv":         ("OT",          10),
    "base_mistral/base_mistral_results_christian.csv":               ("Base Mistral", 0),
}

DATASET_COLORS = {
    "NT":           "#2196F3",
    "OT":           "#FF9800",
    "Full":         "#4CAF50",
    "Base Mistral": "#E53935",
}

DATASET_MARKERS = {
    "NT":           "o",
    "OT":           "s",
    "Full":         "^",
    "Base Mistral": "*",
}

DATASET_LINESTYLES = {
    "NT":           "-",
    "OT":           ":",
    "Full":         "--",
    "Base Mistral": "-",
}


def load_data() -> pd.DataFrame:
    records = []
    missing = []

    for rel_path, (dataset, epochs) in FILE_META.items():
        csv_path = _here / rel_path
        if not csv_path.exists():
            missing.append(rel_path)
            continue

        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                christian = row.get("christian_mistral", "")
                aligned   = row.get("aligned_mistral", "")
                philosophical = row.get("philosophical", "")
                if philosophical == "":
                    continue
                records.append({
                    "dataset":       dataset,
                    "epochs":        epochs,
                    "philosophical": philosophical.strip().lower() in ("true", "1", "yes"),
                    "christian":     float(christian) if christian != "" else None,
                    "aligned":       float(aligned)   if aligned   != "" else None,
                    "question_id":   row.get("question_id", ""),
                })

    if missing:
        print(f"Brakujące pliki (pomiń jeśli score_christian.py jeszcze liczy):")
        for m in missing:
            print(f"  {m}")

    return pd.DataFrame(records)


def draw_row(axes, agg, metric_col, ylabel, all_datasets):
    subtitles = {True: "Pytania filozoficzne", False: "Pytania niefilozoficzne"}
    other_datasets = [d for d in all_datasets if d != "Base Mistral"]

    for ax, is_phil in zip(axes, [True, False]):
        subset = agg[agg["philosophical"] == is_phil].dropna(subset=[metric_col])

        # Base Mistral — pozioma szara linia referencyjna
        base_row = subset[subset["dataset"] == "Base Mistral"]
        if not base_row.empty:
            base_val = base_row[metric_col].iloc[0]
            ax.axhline(
                y=base_val,
                color="#E53935", linewidth=1.5, linestyle="-",
                label="Base Mistral", zorder=1,
            )
            ax.annotate(
                f"{base_val:.1f}",
                (10, base_val),
                textcoords="offset points", xytext=(4, 4),
                ha="left", fontsize=7.5, color="#E53935",
            )

        for dataset in other_datasets:
            ds_data = subset[subset["dataset"] == dataset].sort_values("epochs")
            if ds_data.empty:
                continue
            color  = DATASET_COLORS.get(dataset, "#333333")
            marker = DATASET_MARKERS.get(dataset, "o")
            ax.plot(
                ds_data["epochs"], ds_data[metric_col],
                marker=marker, color=color, linewidth=2, markersize=8, label=dataset,
                linestyle=DATASET_LINESTYLES.get(dataset, "-"), zorder=2,
            )
            for _, row in ds_data.iterrows():
                ax.annotate(
                    f"{row[metric_col]:.1f}",
                    (row["epochs"], row[metric_col]),
                    textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=7.5, color=color,
                )

        ax.set_title(subtitles[is_phil], fontsize=11)
        ax.set_xlabel("Liczba epok trenowania", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks([0, 1, 5, 10])
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
        ax.set_ylim(0, 105)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.grid(axis="y", which="minor", linestyle=":", alpha=0.3)
        ax.legend(title="Zbiór danych", fontsize=9)


def plot(df: pd.DataFrame):
    if df.empty:
        print("Brak danych do wykresu.")
        return

    agg = (
        df.groupby(["dataset", "epochs", "philosophical"])[["christian", "aligned"]]
        .mean()
        .reset_index()
    )

    all_datasets = sorted(agg["dataset"].unique(), key=lambda d: (d == "Base Mistral", d))

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), sharey=False)
    fig.suptitle("Analiza wyników modeli wg liczby epok i typu pytania", fontsize=14, fontweight="bold")

    # wiersz 1 — christian_mistral
    fig.text(0.01, 0.73, "christian_mistral", va="center", rotation="vertical", fontsize=12, fontweight="bold", color="#555")
    draw_row(axes[0], agg, "christian", "Avg christian_mistral (0–100)", all_datasets)

    # wiersz 2 — aligned_mistral
    fig.text(0.01, 0.27, "aligned_mistral", va="center", rotation="vertical", fontsize=12, fontweight="bold", color="#555")
    draw_row(axes[1], agg, "aligned", "Avg aligned_mistral (0–100)", all_datasets)

    plt.tight_layout(rect=[0.03, 0, 1, 1])
    out_path = _here / "score_analysis.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wykres zapisano: {out_path}")
    plt.show()


def print_summary(df: pd.DataFrame):
    if df.empty:
        return
    for metric in ("christian", "aligned"):
        sub = df.dropna(subset=[metric])
        if sub.empty:
            continue
        agg = (
            sub.groupby(["dataset", "epochs", "philosophical"])[metric]
            .agg(["mean", "std", "count"])
            .round(2)
        )
        agg.index = agg.index.set_names(["Dataset", "Epoki", "Filozoficzne"])
        print(f"\n=== {metric}_mistral ===")
        print(agg.to_string())


if __name__ == "__main__":
    df = load_data()
    print(f"Załadowano {len(df)} wierszy z {df['dataset'].nunique()} modeli.\n")
    print_summary(df)
    plot(df)
