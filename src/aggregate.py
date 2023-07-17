from pathlib import Path

import pandas as pd
from tap import Tap

import src.utils as utils


class Args(Tap):
    input_dir: Path = "./outputs"
    output_dir: Path = "./results"


SETTING_COLUMNS = [
    "model_name",
    "lr",
    "lora_r",
    "template_type",
]


def main(args: Args):
    data = []
    for path in args.input_dir.glob("**/test-metrics.json"):
        test_metrics = utils.load_json(path)
        val_metrics = utils.load_json(path.parent / "val-metrics.json")
        config = utils.load_json(path.parent / "config.json")

        data.append(
            {
                "model_name": config["model_name"],
                "lr": config["lr"],
                "lora_r": config["lora_r"],
                "template_type": config["template_type"],
                "best-val-epoch": val_metrics["best-epoch"],
                "best-val-f1": val_metrics["f1"],
                **test_metrics,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = (
        pd.DataFrame(data)
        .groupby(SETTING_COLUMNS, as_index=False)
        .apply(lambda x: x.head(1))
        .reset_index(drop=True)
        .sort_values("f1", ascending=False)
    )
    df.to_csv(str(args.output_dir / "all.csv"), index=False)

    best_model_df = (
        df[df["lora_r"] == 32]
        .groupby("model_name", as_index=False)
        .apply(lambda x: x.nlargest(1, "best-val-f1").reset_index(drop=True))
        .reset_index(drop=True)
    ).sort_values("f1", ascending=False)
    best_model_df.to_csv(str(args.output_dir / "best.csv"), index=False)

    print("-" * 80)
    for row in best_model_df.to_dict("records"):
        print(
            f'|[{row["model_name"]}](https://huggingface.co/{row["model_name"]})|{row["accuracy"]*100:.2f}|{row["precision"]*100:.2f}|{row["recall"]*100:.2f}|{row["f1"]*100:.2f}|'
        )

    best_template_df = (
        df[(df["lora_r"] == 32)]
        .groupby(["model_name", "template_type"], as_index=False)
        .apply(lambda x: x.nlargest(1, "best-val-f1").reset_index(drop=True))
        .reset_index(drop=True)
    ).sort_values(["model_name", "best-val-f1"], ascending=False)
    best_template_df.to_csv(str(args.output_dir / "template.csv"), index=False)
    print("-" * 80)
    for row in best_template_df.to_dict("records"):
        print(
            f'|[{row["model_name"]}](https://huggingface.co/{row["model_name"]})|{row["template_type"]}|{row["best-val-f1"]*100:.2f}|{row["f1"]*100:.2f}|'
        )

    best_lr_df = df[
        (df["lora_r"] == 32)
        & (df["model_name"] == "rinna/japanese-gpt-neox-3.6b")
        & (df["template_type"] == 2)
    ].sort_values("lr", ascending=False)
    best_lr_df.to_csv(str(args.output_dir / "lr.csv"), index=False)
    print("-" * 80)
    for row in best_lr_df.to_dict("records"):
        print(
            f'|{row["lr"]:e}|{row["best-val-f1"]*100:.2f}|{row["accuracy"]*100:.2f}|{row["precision"]*100:.2f}|{row["recall"]*100:.2f}|{row["f1"]*100:.2f}|'
        )

    best_r_df = (
        df[(df["model_name"] == "rinna/japanese-gpt-neox-3.6b") & (df["template_type"] == 2)]
        .groupby("lora_r", as_index=False)
        .apply(lambda x: x.nlargest(1, "best-val-f1").reset_index(drop=True))
        .reset_index(drop=True)
    ).sort_values("best-val-f1", ascending=False)
    best_r_df.to_csv(str(args.output_dir / "r.csv"), index=False)
    print("-" * 80)
    for row in best_r_df.to_dict("records"):
        print(
            f'|{row["lora_r"]}|{row["lr"]:e}|{row["best-val-f1"]*100:.2f}|{row["accuracy"]*100:.2f}|{row["precision"]*100:.2f}|{row["recall"]*100:.2f}|{row["f1"]*100:.2f}|'
        )


if __name__ == "__main__":
    args = Args().parse_args()
    main(args)
