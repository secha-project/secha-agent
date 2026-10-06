import ggsql
import pandas as pd
import pytest

from scripts.probe_visualization_review import bar_values


@pytest.mark.parametrize("projection", ["", "PROJECT y, x TO cartesian"])
def test_probe_compares_values_independently_of_bar_orientation(projection):
    frame = pd.DataFrame({"site": ["East", "West"], "power": [12, 27]})
    query = f"VISUALISE site AS x, power AS y DRAW bar {projection}"
    spec = ggsql.render_altair(frame, query).to_dict()

    assert bar_values(spec) == (("East", 12), ("West", 27))

    frame["power"] = [27, 12]
    swapped = ggsql.render_altair(frame, query).to_dict()
    assert bar_values(swapped) != bar_values(spec)
