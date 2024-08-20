"""
Analyzes commercial airflight data from an online location
"""

import os
import sys
import zipfile
from typing import Optional
from pydantic import BaseModel
import pandas as pd
import requests
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import openai
from openai import OpenAI

SCRIPTS_PATH = os.path.dirname(__file__)
PARENT_PATH = os.path.dirname(SCRIPTS_PATH)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
sys.path.append(SCRIPTS_PATH)
import function_calculate_distance as fcd


class FlightDataAnalyzer(BaseModel):
    """
    Initilizes a tool for analyzing commercial airflight data to support
    sustainability studies.

    When the class is called, the flight data is automatically downloaded
    and stored into pandas dataframes (after removing superfluous columns).

    Attributes
    ----------
    airlines: pd.DataFrame
        Dataframe with airline information such as Name, IATA and ICAO
        codes, Country, Active status, etc.
    airplanes: pd.DataFrame
        Dataframe with Name and IATA and ICAO codes of airplanes.
    airports: pd.DataFrame
        Dataframe with airport information, such as Name, City, Country,
        IATA and ICAO codes, Latitude & Longitude, Timezone, etc.
    routes: pd.DataFrame
        Dataframe with route information, such as Airline, Source Airport,
        Destination Airport, number of Stops, etc.

    Methods
    --------
    plot_airports_map()
        Plots a map with the locations of airports in the specified country.
    distance_analysis()
        Plots the distribution of flight distances for all flights.
    plot_flights_from_airport()
        Plots a map with the flights departing from the specified airport.
    plot_most_used_airplane_models()
        Plots the N most used airplane models by number of routes.
    plot_flights_from_country()
        Plots a map with the flights departing from the specified airport.
    calculate_short_haul_routes()
        Calculates the short-haul routes for a given country.
    aircrafts()
        Prints only the aircraft names
    aircraft_info()
        Prints the aircraft information for the given aircraft name
    airport_info()
        Prints the airport information for the given airport name
    get_most_similar_aircraft()
        Returns the most similar aircraft name based on the input name
    calculate_similarity()
        Returns the similarity between two strings based on the shared characters
    """

    airlines: Optional[pd.DataFrame] = None
    airplanes: Optional[pd.DataFrame] = None
    airports: Optional[pd.DataFrame] = None
    routes: Optional[pd.DataFrame] = None

    class Config:
        """
        Configuration class to change the behavior of Pydantic's BaseModel.
        Set arbitrary_types_allowed to True to allow pandas dataframes to be
        stored as attributes.
        """

        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not OPENAI_API_KEY:
            raise ValueError(
                "API key is not set. Please set the OPENAI_API_KEY environment variable."
            )
        openai.api_key = OPENAI_API_KEY
        self._download()

    def _download(self):
        """
        Downloads the commercial airflight datasets into downloads/
        directory, and stores it into pandas dataframes. If data is
        already downloaded, it won't be redownloaded. Automatically
        called when class is initialized. All cleans the data to use the
        IACO code in the routes dataframe as IATA has many missing values.
        To finish the routes df, it will also calculate the distance between
        the source and destination airports and add it as a new column using
        our newly defined distance_to function in a seperate file.
        Removes all NaN values, as a user cannot work with these at all.

        Parameters
        -----------
        self: class
            The FlightDataAnalyzer class itself.

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
            DATA_URL = (
                "https://gitlab.com/adpro1/adpro2024/-/raw/main/Files/"
                "flight_data.zip?inline=false"
            )
            try:
                response = requests.get(DATA_URL, timeout=15)
                response.raise_for_status()  # raises HTTP error for codes 400 to 600

                # Save downloaded file
                with open(DATA_PATH, "wb") as f:
                    f.write(response.content)
                    print(f"Data downloaded and saved to {DATA_DIRECTORY}")

            except requests.RequestException as error:
                print(f"Failed to download data: {error}")
        else:
            print("Data already downloaded")

        # Load the data into pandas dataframes & store useful columns in class attributes
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

        # Drop CodeShare of Routes since its only NaN
        self.routes = self.routes.drop("Codeshare", axis=1)

        # Merge route airport with airport dataframe from Airport ID or
        # IATA code to get all airport information
        self.routes = pd.merge(
            self.routes,
            self.airports,
            left_on="Source airport",
            right_on="IATA",
            how="left",
        )
        self.routes = pd.merge(
            self.routes,
            self.airports,
            left_on="Destination airport",
            right_on="IATA",
            how="left",
            suffixes=("_Source", "_Destination"),
        )

        # Drop superfluous columns
        self.routes = self.routes.drop(
            [
                "Country_Source",
                "Country_Destination",
                "IATA_Source",
                "IATA_Destination",
                "Latitude_Destination",
                "Longitude_Destination",
                "Altitude_Destination",
                "Timezone_Destination",
                "DST_Destination",
                "Tz database time zone_Destination",
                "City_Destination",
                "Airport ID_Destination",
                "Latitude_Source",
                "Longitude_Source",
                "Altitude_Source",
                "Timezone_Source",
                "DST_Source",
                "Tz database time zone_Source",
                "City_Source",
                "Airport ID_Source",
            ],
            axis=1,
        )

        # Change columns index
        self.routes = self.routes.reindex(
            columns=[
                "Airline",
                "Airline ID",
                "Source airport",
                "Source airport ID",
                "ICAO_Source",
                "Name_Source",
                "Destination airport",
                "Destination airport ID",
                "ICAO_Destination",
                "Name_Destination",
                "Stops",
                "Equipment",
            ]
        )

        # Drop NaN values
        self.airlines.dropna(inplace=True)
        self.airplanes.dropna(inplace=True)
        self.airports.dropna(inplace=True)
        self.routes.dropna(inplace=True)

        # Make sure all airports specified in routes also exist in airports for continuity
        self.routes = self.routes[
            self.routes["ICAO_Source"].isin(self.airports["ICAO"])
        ]
        self.routes = self.routes[
            self.routes["ICAO_Destination"].isin(self.airports["ICAO"])
        ]

        # Add the distance and save it as specified by Day 1, Phase 2
        self.routes["Distance"] = self.routes.apply(
            lambda row: fcd.distance_to(
                self, row["ICAO_Source"], row["ICAO_Destination"]
            ),
            axis=1,
        )

    def plot_airports_map(self, country):
        """
        Plots a map with the locations of airports in the specified country.

        Parameters
        ----------
        self : class
            The FlightDataAnalyzer class itself.
        country : str
            The name of the country.

        Returns
        -------
        Nothing. Plots map.
        """

        # Check if the country exists in the airports dataframe
        if country not in self.airports["Country"].unique():
            raise ValueError(f"Error: Country '{country}' does not exist.")

        # Filter airports dataframe by country
        country_airports = self.airports[self.airports["Country"] == country]

        fig = go.Figure()
        country_airports.fillna("", inplace=True)

        # Convert data to lists
        cities = country_airports["Name"].astype(str).tolist()
        countries = country_airports["Country"].astype(str).tolist()

        # Combine city and country information
        scatter_hover_data = [
            country + " : " + city for city, country in zip(cities, countries)
        ]

        fig.add_trace(
            go.Scattergeo(
                lon=country_airports["Longitude"].values.tolist(),
                lat=country_airports["Latitude"].values.tolist(),
                hoverinfo="text",
                text=scatter_hover_data,
                mode="markers",
                marker={
                    "size": 10,
                    "color": 'blue', 
                    "opacity": 0.1
                },
            )
        )

        fig.update_layout(
            title_text="Airports in " + country + " on a map",
            height=500,
            width=500,
            margin={"t": 0, "b": 0, "l": 0, "r": 0, "pad": 0},
            showlegend=False,
            geo={
                "projection_type": 'orthographic',
                "showland": True,
                "landcolor": 'lightgrey',
                "countrycolor": 'grey',
            },
        )
        fig.show()

    def distance_analysis(self):
        """
        Plots the distribution of flight distances for all flights.

        Parameters
        ----------
        self : class
            The FlightDataAnalyzer class itself.

        Returns
        -------
        Nothing. Plots histogram with distance distribution.
        """
        # Remove duplicate routes (A to B is the same as B to A)
        self.routes["Route"] = self.routes.apply(
            lambda row: frozenset([row["ICAO_Source"], row["ICAO_Destination"]]), axis=1
        )
        unique_routes = self.routes.drop_duplicates(subset=["Route"])
        # Plot the histogram
        plt.hist(unique_routes["Distance"], bins=15)
        plt.xlabel("Flight Distance")
        plt.ylabel("Frequency")
        plt.title("Distribution of Flight Distances")
        plt.show()

    def plot_flights_from_airport(
        self, airport, internal=False, cutoff_distance=1000, limit=200
    ):
        """
        Plots a map with the flights departing from the specified airport.
        If internal is True, only flights with destination in the same country
        as the departure airport will be plotted. If too many flights are found,
        only the 200 largest routes will be plotted. The color of the lines will
        be red for short-haul flights (distance <= cutoff_distance) and orange
        for long-haul flights.
        
        Parameters
        ----------
        airport : str
            The name of the departure airport.
        internal : bool, optional
            Whether to plot only internal flights or all flights. Default is False.
        cutoff_distance: int, optional
            The maximum distance for a route to be considered short-haul. Default is 1000.
        limit : int, optional
            The number of routes to plot. Default is 200.
        Returns
        -------
        Nothing. Plots map.
        """
        limited = False
        # Check if the airport exists in the airports dataframe
        if airport not in self.airports["Name"].unique():
            print(f"Error: Airport '{airport}' does not exist.")
            return

        # Get start airport data
        start_airport = self.airports[self.airports["Name"] == airport]
        start_iata = start_airport["ICAO"].values[0]

        # Filter routes dataframe by departure airport
        departures = self.airports[self.airports["Name"] == airport]
        departures_merged = pd.merge(
            departures, self.routes, left_on="ICAO", right_on="ICAO_Source", how="left"
        )
        departures = departures_merged["ICAO_Destination"]
        departures = pd.merge(
            departures,
            self.airports,
            left_on="ICAO_Destination",
            right_on="ICAO",
            how="left",
        )

        # Filter routes dataframe by destination airport
        if internal:
            country = self.airports[self.airports["Name"] == airport]["Country"].values[
                0
            ]
            departures = departures[departures["Country"] == country]

        departures["Distance"] = departures.apply(
            lambda row: fcd.distance_to(self, start_iata, row["ICAO_Destination"]),
            axis=1,
        )

        # If we have too many departures just show the 50 longest routes
        if len(departures) > limit:
            # Remove NaN airports
            departures = departures.nlargest(limit, "Distance")
            limited = True

        fig = go.Figure()
        start_coords = (
            start_airport["Longitude"].values[0],
            start_airport["Latitude"].values[0],
        )  # Extract values
        start_lon = start_coords[0]
        start_lat = start_coords[1]
        for _, row in departures.iterrows():
            color = "red" if row["Distance"] <= cutoff_distance else "orange"
            fig.add_trace(
                go.Scattergeo(
                    locationmode="ISO-3",
                    lon=[start_lon, row["Longitude"]],
                    lat=[start_lat, row["Latitude"]],
                    mode="lines",
                    line={"width": 1, "color": color},
                    opacity=0.5,
                )
            )

        scatter_hover_data = [
            f"{row['Country']} : {row['Name']}" for _, row in departures.iterrows()
        ]

        fig.add_trace(
            go.Scattergeo(
                lon=departures["Longitude"].values.tolist(),
                lat=departures["Latitude"].values.tolist(),
                hoverinfo="text",
                text=scatter_hover_data,
                mode="markers",
                marker={
                    "size": 10,
                    "color": 'grey',
                    "opacity": 0.1
                },
            )
        )

        # Convert NaN values to empty strings for city and country columns
        departures["ICAO_Destination"] = departures["ICAO_Destination"].fillna("")
        departures["Country"] = departures["Country"].fillna("")

        # Convert data to lists
        cities = (
            start_airport["Name"].astype(str).tolist()
            + departures["Name"].astype(str).tolist()
        )
        countries = (
            start_airport["Country"].astype(str).tolist()
            + departures["Country"].astype(str).tolist()
        )

        # Combine city and country information
        scatter_hover_data = [
            country + " : " + city for city, country in zip(cities, countries)
        ]

        fig.add_trace(
            go.Scattergeo(
                lon=start_airport["Longitude"].values.tolist()
                + departures["Longitude"].values.tolist(),
                lat=start_airport["Latitude"].values.tolist()
                + departures["Latitude"].values.tolist(),
                hoverinfo="text",
                text=scatter_hover_data,
                mode="markers",
                marker={
                    "size": 10,
                    "color": 'blue',
                    "opacity": 0.1
                },
            )
        )
        title_text_unit = ""
        if limited:
            if internal:
                title_text_unit = (f"Internal Flights from {airport} "
                                   f"(limited to {limit} longest routes)")
            else:
                title_text_unit = (
                    f"All Flights from {airport} (limited to {limit} longest routes)"
                )
        else:
            if internal:
                title_text_unit = f"Internal Flights from {airport}"
            else:
                title_text_unit = f"All Flights from {airport}"
        ## Update graph layout to improve graph styling.
        fig.update_layout(
            title_text=title_text_unit,
            height=500,
            width=500,
            margin={"t": 0, "b": 0, "l": 0, "r": 0, "pad": 0},
            showlegend=False,
            geo={
                "projection_type": 'orthographic',
                "showland": True,
                "landcolor": 'lightgrey',
                "countrycolor": 'grey',
            },
        )

        fig.show()

    def plot_most_used_airplane_models(self, countries=None, n=10):
        """
        Plots the N most used airplane models by number of routes.
        If countries is None, it plots for all dataset. If countries
        is a string or list of strings, it plots just for that subset.

        Parameters
        ----------
        countries : str or list of str, optional
            The name of the country or list of countries. Default is None.
        N : int, optional
            The number of airplane models to plot. Default is 5.

        Returns
        -------
        Nothing. Plots map.
        """

        # Filter routes dataframe by countries if provided
        if countries is not None:
            if isinstance(countries, str):
                countries = [countries]
            routes_filtered = pd.merge(
                self.routes,
                self.airports,
                left_on="Source airport",
                right_on="IATA",
                how="left",
            )
            routes_filtered = routes_filtered[
                routes_filtered["Country"].isin(countries)
            ]
        else:
            routes_filtered = self.routes

        # Group routes by airplane model and count the number of routes
        airplane_models = routes_filtered.groupby("Equipment").size().nlargest(n)

        # Plot the bar chart
        plt.bar(airplane_models.index, airplane_models.values)
        plt.xlabel("Airplane Model")
        plt.ylabel("Number of Routes")
        plt.title(f"Top {n} Most Used Airplane Models")
        plt.xticks(rotation=45)
        plt.show()

    def plot_flights_from_country(
        self, country, internal=False, cutoff_distance=1000, limit=200
    ):
        """
        Plots a map with the flights departing from all airports in the specified country.
        If internal is True, only internal flights (within the same country) will be plotted.
        If too many flights are found, only the 200 largest routes will be plotted.
        The color of the lines will be red for short-haul flights (distance <= cutoff_distance)
        and orange for long-haul flights.

        Parameters
        ----------
        country : str
            The name of the departure country.
        internal : bool, optional
            Whether to plot only internal flights or all flights. Default is False.
        cutoff_distance: int, optional
            The maximum distance for a route to be considered short-haul. Default is 1000.
        limit : int, optional
            The number of routes to plot. Default is 200.
        Returns
        -------
        None
        """
        fig = go.Figure()
        limited = False
        # Check if the country exists in the airports dataframe
        if country not in self.airports["Country"].unique():
            print(f"Error: Country '{country}' does not exist.")
            return

        # Filter airports dataframe by country
        airports_in_country = self.airports[self.airports["Country"] == country]

        # Merge with routes
        departures_merged = pd.merge(
            airports_in_country,
            self.routes,
            left_on="ICAO",
            right_on="ICAO_Source",
            how="left",
        )

        # Add destination airport data
        departures = pd.merge(
            departures_merged,
            self.airports,
            left_on="ICAO_Destination",
            right_on="ICAO",
            how="left",
        )

        if internal:
            departures = departures[departures["Country_x"] == departures["Country_y"]]

        departures = departures.dropna(
            subset=[
                "Latitude_x",
                "Longitude_x",
                "Latitude_y",
                "Longitude_y",
                "ICAO_Source",
                "ICAO_Destination",
            ]
        )
        departures["Distance"] = departures.apply(
            lambda row: fcd.distance_to(
                self, row["ICAO_Source"], row["ICAO_Destination"]
            ),
            axis=1,
        )

        if len(departures) > limit:
            departures = departures.nlargest(limit, "Distance")
            limited = True

        for _, row in departures.iterrows():
            color = "red" if row["Distance"] <= cutoff_distance else "orange"
            fig.add_trace(
                go.Scattergeo(
                    locationmode="ISO-3",
                    lon=[row["Longitude_x"], row["Longitude_y"]],
                    lat=[row["Latitude_x"], row["Latitude_y"]],
                    mode="lines",
                    line={"width": 1, "color": color},
                    opacity=0.5,
                )
            )

        # Convert NaN values to empty strings for city and country columns
        departures["ICAO_Source"] = departures["ICAO_Source"].fillna("")
        departures["ICAO_Destination"] = departures["ICAO_Destination"].fillna("")
        departures["Country_x"] = departures["Country_x"].fillna("")
        departures["Country_y"] = departures["Country_y"].fillna("")

        # Convert data to lists
        cities = (
            departures["ICAO_Source"].astype(str).tolist()
            + departures["ICAO_Destination"].astype(str).tolist()
        )
        countries = (
            departures["Country_x"].astype(str).tolist()
            + departures["Country_y"].astype(str).tolist()
        )

        # Combine city and country information
        scatter_hover_data = [
            country + " : " + city for city, country in zip(cities, countries)
        ]

        fig.add_trace(
            go.Scattergeo(
                lon=departures["Longitude_x"].values.tolist()
                + departures["Longitude_y"].values.tolist(),
                lat=departures["Latitude_x"].values.tolist()
                + departures["Latitude_y"].values.tolist(),
                hoverinfo="text",
                text=scatter_hover_data,
                mode="markers",
                marker={
                    "size": 10,
                    "color": 'blue',
                    "opacity": 0.1
                },
            )
        )
        title_text_unit = ""
        if limited:
            if internal:
                title_text_unit = (f"Internal Flights from {country} "
                                   f"(limited to {limit} longest routes)")
            else:
                title_text_unit = (
                    f"All Flights from {country} (limited to {limit} longest routes)"
                )
        else:
            if internal:
                title_text_unit = f"Internal Flights from {country}"
            else:
                title_text_unit = f"All Flights from {country}"

        ## Update graph layout to improve graph styling.
        fig.update_layout(
            title_text=title_text_unit,
            height=500,
            width=500,
            margin={"t": 0, "b": 0, "l": 0, "r": 0, "pad": 0},
            showlegend=False,
            geo={
                "projection_type": 'orthographic',
                "showland": True,
                "landcolor": 'lightgrey',
                "countrycolor": 'grey',
            },
        )

        fig.show()

    def calculate_short_haul_routes(self, country, cutoff_distance=1000):
        """
        Calculates the short-haul routes for a given country.
        Short-haul routes are defined as routes with a distance less 
        than or equal to cutoff_distance.
        The function returns a dataframe with the short-haul routes 
        sorted by distance.

        Parameters
        ----------
        country : str
            The name of the country.
        cutoff_distance : float, optional
            The maximum distance for a route to be considered short-haul. Default is 1000.
        Returns
        -------
        pd.DataFrame
            Dataframe with the short-haul routes sorted by distance.
        """
        # Filter airports dataframe by country
        airports_in_country = self.airports[self.airports["Country"] == country]

        departures = pd.merge(
            airports_in_country, self.routes, left_on="ICAO", right_on="ICAO_Source"
        )
        departures = pd.merge(
            departures,
            airports_in_country,
            left_on="ICAO_Destination",
            right_on="ICAO",
            suffixes=("_source", "_destination"),
        )

        # Calculate distance and filter by cutoff_distance
        departures["Distance"] = departures.apply(
            lambda row: fcd.distance_to(
                self, row["ICAO_source"], row["ICAO_destination"]
            ),
            axis=1,
        )
        short_haul_routes = departures[departures["Distance"] <= cutoff_distance]

        # Remove duplicate routes (A to B is the same as B to A)
        short_haul_routes["Route"] = short_haul_routes.apply(
            lambda row: frozenset([row["ICAO_Source"], row["ICAO_Destination"]]), axis=1
        )
        unique_short_haul = short_haul_routes.drop_duplicates(subset=["Route"])

        num_short_haul_routes = unique_short_haul.shape[0]
        total_short_haul_distance = unique_short_haul["Distance"].sum()

        # Sort short-haul routes by distance for plotting
        sorted_short_haul = unique_short_haul.sort_values(by="Distance")

        # Plot the short haul routes distances
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                y=list(range(len(sorted_short_haul))),
                x=sorted_short_haul["Distance"],
                mode="lines+markers",
                name="Short-haul Route Distances",
            )
        )

        # Using the ballpark figure that trains are at least 12 times
        # more energy efficient per passenger than air travel.
        emission_reduction_factor = (
            12  # This is a simplification for illustrative purposes.
        )

        # Calculate potential emissions savings by replacing flights with trains
        potential_emissions_savings = (
            total_short_haul_distance / emission_reduction_factor
        )
        sorted_short_haul["Emissions Savings (km equivalent)"] = (
            sorted_short_haul["Distance"] / emission_reduction_factor
        )

        # Drop usless columns
        sorted_short_haul = sorted_short_haul.drop(
            [
                "Country_source",
                "Country_destination",
                "ICAO_source",
                "ICAO_destination",
                "Latitude_source",
                "Longitude_source",
                "Altitude_source",
                "Timezone_source",
                "DST_source",
                "Tz database time zone_source",
                "City_source",
                "Airport ID_source",
                "Latitude_destination",
                "Longitude_destination",
                "Altitude_destination",
                "Timezone_destination",
                "DST_destination",
                "Tz database time zone_destination",
                "City_destination",
                "Airport ID_destination",
                "Airline",
                "Airline ID",
                "Source airport ID",
                "ICAO_Source",
                "Destination airport ID",
                "ICAO_Destination",
                "IATA_source",
                "Name_Source",
                "Name_destination",
                "Stops",
                "Equipment",
                "IATA_destination",
            ],
            axis=1,
        )

        # Reindex columns
        sorted_short_haul = sorted_short_haul.reindex(
            columns=[
                "Name_source",
                "Source airport",
                "Name_Destination",
                "Destination airport",
                "Distance",
                "Emissions Savings (km equivalent)",
                "Route",
            ]
        )

        # Add this calculation to the plot annotation
        fig.add_annotation(
            x=0.5,
            y=1.1,
            xref="paper",
            yref="paper",
            text=(f"Total short-haul routes: {num_short_haul_routes}, "
                  f"Total distance: {total_short_haul_distance} km, "
                  f"Potential emissions savings: {potential_emissions_savings:.2f} "
                  f"km equivalent by train"),
            showarrow=False,
            font={"size": 12},
            align="center",
            bgcolor="lightgrey",
            bordercolor="black",
            borderwidth=2,
            borderpad=4,
            opacity=0.8,
        )

        # Update layout to better accommodate the annotation
        fig.update_layout(
            title="Short-Haul Flight Route Distances",
            yaxis_title="Route Index",
            xaxis_title="Distance (km)",
            margin={"t": 120},
        )

        fig.show()

        return sorted_short_haul

    def aircrafts(self):
        """
        Prints only the aircraft names

        Parameters
        ----------
        self : class

        """
        # Print unique names
        print(self.airplanes["Name"].unique())

    # Define a new method called **aircraft_info** that receives a string called _aircraft_name_.
    # If the string is **NOT** in the list of aircrafts in the data, it should return an exception
    # and present a way to guide the user into how they could choose a correct aircraft name.

    def aircraft_info(self, aircraft_name):
        """
        Prints the aircraft information for the given aircraft name
        If it does not exist, it throws and error and calculates the 
        most similar aircraft name based on the input and suggests it to
        the user

        Parameters
        ----------
        self : class
        aircraft_name : str
            The name of the aircraft
        """
        # Check if the aircraft exists
        if aircraft_name not in self.airplanes["Name"].unique():
            # Get most similiar string based on input from aircrafts list
            most_similar = self.get_most_similar_aircraft(aircraft_name)
            raise ValueError(
                f"The aircraft name {aircraft_name} does not exist. Did you mean: {most_similar}"
            )
        else:
            client = OpenAI(api_key=OPENAI_API_KEY)
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",  # Use an appropriate model
                messages=[
                    {
                        "role": "system",
                        "content": (f"You are a airplane expert that can provide "
                                    f"specifications for {aircraft_name}."),
                    },
                    {
                        "role": "user",
                        "content": f"What are the specifications of {aircraft_name}?",
                    },
                ],
            )
        # Extract specifications from the API response (replace with actual parsing logic)
        # Format specifications in Markdown table format
        specifications = completion.choices[0].message
        # Format specifications in Markdown table format
        if hasattr(specifications, "content"):
            specifications_content = specifications.content
            specifications_table = f"| Information | {aircraft_name} |\n \n"
            lines = specifications_content.split("<br>")
            first_line = lines[0].strip()
            specifications_table += (
                f"| {first_line} | |\n"  # First line as a single row
            )

            for line in lines[1:]:
                if line.strip():  # Skip empty lines
                    spec_key, spec_value = line.split(":", 1)
                    specifications_table += (
                        f"| {spec_key.strip()} | {spec_value.strip()} |\n"
                    )
        else:
            specifications_table = "| Specifications | No specifications available |\n"

        # Print the formatted specifications

        print(specifications_table)
        print("\n| --------- | ----- |\n")

    def get_most_similar_aircraft(self, input_name):
        """
        Returns the most similar aircraft name based on the input name

        Parameters
        ----------
        self : class
        input_name : str
            The name of the aircraft
        Returns
        -------
        str
            The most similar aircraft name
        """
        max_similarity = 0
        most_similar = None
        aircraft_name = self.airplanes["Name"].unique()
        for aircraft in aircraft_name:
            similarity = self.calculate_similarity(input_name, aircraft)
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar = aircraft

        return most_similar

    def calculate_similarity(self, input_name, target_name):
        """
        Returns the similarity between two strings based on the shared characters

        Parameters
        ----------
        self : class
        input_name : str
            The name of the aircraft
        target_name : str
            The other aircraft name
        Returns
        -------
        float
            The similarity between the two strings
        """
        # Simple similarity calculation based on shared characters
        shared_chars = set(input_name) & set(target_name)
        similarity = len(shared_chars) / max(len(input_name), len(target_name))
        return similarity

    def airport_info(self, airport_name):
        """
        Prints the airport information for the given airport name
        No error checking as not needed in task statement.

        Parameters
        ----------
        self : class
        airport_name : str
            The name of the airport
        """
        # print(self.OPENAI_API_KEY)
        client = OpenAI(api_key=OPENAI_API_KEY)
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Use an appropriate model
            messages=[
                {
                    "role": "system",
                    "content": (f"You are a airport expert that can provide "
                                f"specifications for {airport_name}."),
                },
                {
                    "role": "user",
                    "content": f"What are the specifications of {airport_name}?",
                },
            ],
        )
        # Format specifications in Markdown table format
        specifications = completion.choices[0].message
        # Format specifications in Markdown table format
        if hasattr(specifications, "content"):
            specifications_content = specifications.content
            specifications_table = f"| Information | {airport_name} |\n \n"
            lines = specifications_content.split("<br>")
            first_line = lines[0].strip()
            specifications_table += (
                f"| {first_line} | |\n"  # First line as a single row
            )

            for line in lines[1:]:
                if line.strip():  # Skip empty lines
                    spec_key, spec_value = line.split(":", 1)
                    specifications_table += (
                        f"| {spec_key.strip()} | {spec_value.strip()} |\n"
                    )
        else:
            specifications_table = "| Specifications | No specifications available |\n"

        print(specifications_table)
        print("\n| --------- | ----- |\n")
