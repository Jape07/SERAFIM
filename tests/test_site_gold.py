"""The published site gets simulated gold for every challenge target."""

import pyarrow.parquet as pq

from serafim import contract
from serafim.config import load_config
from serafim.gold.mock import write_all_targets


def test_published_gold_includes_every_target(tmp_path):
    config = load_config()
    gold = tmp_path / "gold"
    write_all_targets(gold, config)
    contract.validate_gold(gold)

    ids = [target["id"] for target in config["targets"]]
    targets = pq.read_table(gold / "targets.parquet").to_pylist()
    assert [row["target_id"] for row in targets] == ids

    # Every target has the same shape: 17 detector-season cells with 3 frames each.
    frames = pq.read_table(gold / "frames.parquet").to_pandas()
    sums = pq.read_table(gold / "sums.parquet").to_pandas()
    assert frames.groupby("target_id").size().to_dict() == dict.fromkeys(ids, 51)
    assert set(sums.target_id) == set(ids)
    for target_id in ids:
        assert (gold / contract.sum_png(target_id, 3, 1)).is_file()
