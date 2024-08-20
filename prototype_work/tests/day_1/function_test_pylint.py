"""
This module defines three tests for the distance_to function defined in
the function_calculate_distance module.
"""

import pytest
from class_flight_data import FlightDataAnalyzer
from function_calculate_distance import distance_to


def test_continent():
    """Test distant calculation for airports in different continents."""
    analyzer = FlightDataAnalyzer()
    airport_oceania = "AYGA"  # Oceania
    airport_asia = "ULDA"  # Asia/Europe (in Russia)
    expected_distance = 10750.81
    assert round(distance_to(analyzer, airport_oceania, airport_asia)) == round(
        expected_distance
    )


def test_error_code():
    """Test if function identifies invalid ICAO codes."""
    analyzer = FlightDataAnalyzer()
    airport_1 = "AYGA"
    airport_invalid = "invalid_code"
    with pytest.raises(ValueError):
        distance_to(analyzer, airport_1, airport_invalid)


def test_same():
    """Test if function can handle airports with same location."""
    analyzer = FlightDataAnalyzer()
    airport_1 = "AYGA"
    assert distance_to(analyzer, airport_1, airport_1) == 0
