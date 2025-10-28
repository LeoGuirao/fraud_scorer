from __future__ import annotations

from datetime import datetime

from fraud_scorer.parsers.gps_direct_extractor import normalize_gps_tables


def test_normalize_gps_tables_recovers_timestamp_coordinates_from_raw():
    table = {
        "headers": [],
        "data_rows": [
            [
                "1 ECO 010",
                "351777096022718 R83",
                "",
                "",
                "",
                "",
                "Encendido",
                "2024-02-13 19:15:00",
                "0 Noreste(Dirección48)",
                "GPS",
                "",
                "8 Verdadero",
                "22.570000/-100.630000",
            ]
        ],
        "gps_plugin": "pdf",
    }

    dataframe, warnings = normalize_gps_tables([table])

    assert warnings == []
    assert dataframe.shape[0] == 1

    timestamp = dataframe.loc[0, "timestamp"]
    assert isinstance(timestamp, datetime)
    assert timestamp.year == 2024
    assert timestamp.month == 2
    assert timestamp.day == 13
    assert timestamp.hour == 19
    assert timestamp.minute == 15

    lat = dataframe.loc[0, "latitude"]
    lon = dataframe.loc[0, "longitude"]
    assert round(float(lat), 3) == 22.57
    assert round(float(lon), 3) == -100.63

    assert dataframe.loc[0, "event_label"] == "Encendido"
