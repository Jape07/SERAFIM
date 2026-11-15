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

    # Until the new contract (task D2), each target is written as one region.
    regions = pq.read_table(gold / "regions.parquet").to_pylist()
    frames = pq.read_table(gold / "frames.parquet")
    ids = [row["region_id"] for row in regions]
    assert ids == [target["id"] for target in config["targets"]]
    assert frames.num_rows == 10 * len(ids)
