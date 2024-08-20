"""
Analyzes commercial airflight data from an online location
"""

import os
import requests
import zipfile
import pandas as pd
from typing import Optional
from pydantic import BaseModel, Field, field_validator

PARENT_PATH = os.path.dirname(os.getcwd())


class FlightDataAnalyzer(BaseModel):
    """
    Initilizes a tool for analyzing commercial airflight data to support
    sustainability studies. When the class is called, the flight data is
    automatically downloaded and stored into pandas dataframes (after
    removing superfluous columns).

    Attributes
    ----------
    airlines: pd.DataFrame
        Dataframe with airline information such as Name, IATA and ICAO
        codes, Country, Active status, etc
    airplanes: pd.DataFrame
        Dataframe with Name and IATA and ICAO codes of airplanes
    airports: pd.DataFrame
        Dataframe with airport information, such as Name, City, Country,
        IATA and ICAO codes, Latitude & Longitude, Timezone, etc.
    routes: pd.DataFrame
        Dataframe with route information, such as Airline, Source Airport,
        Destination Airport, number of Stops, etc.

    Methods
    --------
    ..()
        ..
    """

    airlines: Optional[pd.DataFrame] = None
    airplanes: Optional[pd.DataFrame] = None
    airports: Optional[pd.DataFrame] = None
    routes: Optional[pd.DataFrame] = None

    class Config:  # Include this to allow pandas dataframes as attributes
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._download()

    def _download(self):
        """
        Downloads the commercial airflight datasets into downloads/
        directory, and stores it into pandas dataframes. If data is
        already downloaded, it won't be redownloaded. Automatically
        called when class is initialized.

        Parameters
        -----------
        self: class
            The FlightDataAnalyzer class itself

        Returns
        --------
        Nothing. Defines attributes for the FlightDataAnalyzer, one for
        each dataset:
        - self.airlines: airlines.csv
        - self.airplanes: airplanes.csv
        - self.airports: airports.csv
        - self.routes: routes.csv
        """
        # Check if the directory exists
        DATA_DIRECTORY = os.path.join(PARENT_PATH, "downloads")
        if not os.path.exists(DATA_DIRECTORY):
            os.makedirs(os.path.join(PARENT_PATH, "downloads"))
            print("Created downloads directory")

        # Check if the .zip file exists
        DATA_PATH = os.path.join(DATA_DIRECTORY, "flight_data.zip")
        if not os.path.isfile(DATA_PATH):
            DATA_URL = "https://gitlab.com/adpro1/adpro2024/-/raw/main/Files/flight_data.zip?inline=false"
            try:
                response = requests.get(DATA_URL)
                response.raise_for_status()  # raises HTTP error for codes 400 to 600

                # Save downloaded file
                with open(DATA_PATH, "wb") as f:
                    f.write(response.content)
                    print(f"Data downloaded and saved to {DATA_DIRECTORY}")

            except requests.RequestException as error:
                print(f"Failed to download data: {error}")
        else:
            print("Data already downloaded")

        # Load the data into pandas dataframes and store the useful columns in class attributes
        with zipfile.ZipFile(DATA_PATH, "r") as z:
            with z.open("airlines.csv") as f:
                self.airlines = pd.read_csv(f, index_col=0).drop("Alias", axis=1)

            with z.open("airplanes.csv") as f:
                self.airplanes = pd.read_csv(f, index_col=0)

            with z.open("airports.csv") as f:
                self.airports = pd.read_csv(f, index_col=0).drop(
                    ["Type", "Source"], axis=1
                )

            with z.open("routes.csv") as f:
                self.routes = pd.read_csv(f, index_col=0)
