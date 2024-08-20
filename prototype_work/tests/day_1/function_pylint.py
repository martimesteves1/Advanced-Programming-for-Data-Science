"""
This module provides a function that calculates the distance between two
airports.
"""

import math


def distance_to(analyzer, airport_1: str, airport_2: str) -> float:
    """
    Calculates the real distance in kilometers between 'airport_1' and
    'airport_2', based on information stored in analyzer

    Parameters
    ---------------
    analyzer:
        Instance of FlightDataAnalyzer()
    airport_1: string
        ICAO code of the first airport.
    airport_1: string
        ICAO code of the second airport.

    Returns
    ---------------
    distance: float
        The real distance in kilometers between 'airport_1' and
        'airport_2'.
    """
    df_airport_1 = analyzer.airports[analyzer.airports["ICAO"] == airport_1]
    if df_airport_1.empty:
        raise ValueError(f"Airport code '{airport_1}' not found in airports dataset.")

    df_airport_2 = analyzer.airports[analyzer.airports["ICAO"] == airport_2]
    if df_airport_2.empty:
        raise ValueError(f"Airport code '{airport_2}' not found.")

    # Convert latitude and longitude to radians
    lat_1 = math.radians(float(df_airport_1["Latitude"].iloc[0]))
    lon_1 = math.radians(float(df_airport_1["Longitude"].iloc[0]))
    lat_2 = math.radians(float(df_airport_2["Latitude"].iloc[0]))
    lon_2 = math.radians(float(df_airport_2["Longitude"].iloc[0]))

    # Earth radius in kilometers
    RADIUS = 6371

    # Haversine formula to calculate distance
    dlat = lat_2 - lat_1
    dlon = lon_2 - lon_1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat_1) * math.cos(lat_2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = RADIUS * c

    return distance
