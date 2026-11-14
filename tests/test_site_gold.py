"""The published site gets simulated gold for every watched region."""

import pyarrow.parquet as pq

from serafim import contract
from serafim.config import load_config
from serafim.gold.mock import write_all_regions


def test_published_gold_includes_every_region(tmp_path):
    config = load_config()
    gold = tmp_path / "gold"
    write_all_regions(gold, config)
    contract.validate_gold(gold)

    regions = pq.read_table(gold / "regions.parquet").to_pylist()
    frames = pq.read_table(gold / "frames.parquet")
    ids = [row["region_id"] for row in regions]
    assert ids == [region["id"] for region in config["regions"]]
    assert frames.num_rows == 10 * len(ids)
