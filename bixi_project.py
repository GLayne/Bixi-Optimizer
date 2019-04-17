#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Travail de Session
Algorithmes pour l'optimisation et l'analyse des mégadonnées
HEC Montréal, Hiver 2019

Par: 
Gabriel Lainesse, 11189782
Audrey Leduc, 11258259
Bryan Parahy, 11195258

"""
from cycler import cycler  # used to cycle between colors when plotting with matplotlib at the end of the script
import os  # used to deal with files and paths
import sys  # used  to exit the script if the JSON import fails twice.
import json  # used to parse the JSON data from Bixi into a dictionary
import math  # for math functions, such as floor, sin, cos, tan, sqrt, etc.
import numpy as np  # used to create *real* arrays (instead of lists)
import requests  # used to connect to the web and download the Bixi stations JSON data
import random  # used to randomly assign an initial bike value to a station during initialization
import datetime  # used to name log files
import csv  # used to write a log of the solution as a CSV file
import matplotlib.pyplot as plt  # used to plot the solution at the end of the script
import pandas as pd  # not used directly, but required by geopandas
import geopandas as gpd  # used to plot the island of Montreal underneath the matplotlib plot


class Station:
    """
    A Bixi Station object.
    
    Initial Attributes:
        id_ : The ID number of the station. Used to identify stations.
        name_ : 
        short_name : 
        max_load : A positive integer representing the bike capacity of the station.
        latitude : A signed floating point number representing the latitude.
        longitude : A signed floating point number representing the longitude.
    
    Additional Attributes:
        Coordinates : A tuple combining the latitude and longitude, in that order.
    
    
    Methods:
        calculate_distance

    """
    def __init__(self, id_, bixi_id, name, short_name, max_load, latitude, longitude):
        self.bixi_id = bixi_id
        self.id_ = int(id_) # coerce to int
        self.name = name
        self.short_name = short_name
        self.max_load = max_load
        self.latitude = latitude
        self.longitude = longitude
        self.coordinates = (self.latitude, self.longitude)
        self.current_load = 0 # initialize the current number of bikes as 0
        self.urgency = 0
        self.optimal_load = math.floor((self.max_load / 2))  # station optimal load is half empty-half full (hypothesis)
        self.update_urgency()  # Update urgency value
        
    def __getitem__(self, item):
        return self.Station[item]
        
    def calculate_distance(self, to_station):
        """
        Calculate the distance in kilometers between the current station and the specified station.
        Based on the formula shown on this page: https://www.movable-type.co.uk/scripts/latlong.html
    
         Parameters:
         to_station: Station object which contains a coordinates tuple value
        """
        assert isinstance(to_station, Station), "to_station must be a Station object!"
        assert isinstance(to_station.coordinates, tuple), "the coordinates element of the to_station object must be a \
                tuple in the form (latitude, longitude)!"

        EARTH_RADIUS = 6371.3  # Earth's mean radius
    
        # unpacking coordinate tuples
        lat1, lon1 = self.coordinates
        lat2, lon2 = to_station.coordinates
    
        # storing radian values
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        # calculating deltas
        delta_lat = math.radians(lat2-lat1)
        delta_lon = math.radians(lon2-lon1)
    
        # using the Haversine formula: compute the "as the crow flies" distance approximation,
        # accounting for the Earth's curvature
        a = math.sin(delta_lat/2) * math.sin(delta_lat/2) + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2) * math.sin(delta_lon/2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
        d = EARTH_RADIUS * c
        return d

    def update_urgency(self):  # Update urgency
        if self.max_load == 0:  # Prevents division by 0 for stations without any capacity.
            self.urgency = 0
        else:
            self.urgency = -(self.current_load - self.optimal_load) / self.max_load
            # calculates how far we are from the median, normalized by the max_load


class Truck:
    """
    A truck object, implementing a method of moving bikes around.
    """
    
    def __init__(self, max_load, current_station):
        assert isinstance(max_load, int), "max_load must be an int value!"
        assert max_load > 0, "max_load must be > 0!"
        assert isinstance(current_station, Station), "current_station must be a Station object!"
        
        self.max_load = max_load
        self.current_load = math.floor(max_load / 2)  # initialize truck with an initial load of capacity / 2 (median)
        self.current_station = current_station
        self.available_space = self.max_load - self.current_load
        
    def __getitem__(self, item):
        return self.Truck[item]

    def load_unload_bikes(self, station):
        """
        Takes or put bikes to the station
        
        Arguments:
            station: Station object for the station from which to take or put bikes.
            
        Returns:
            A tuple:
            - int : Number of bikes that were taken, attempting to return the Station to its optimal_load.
            - Station object: updated copy of the Station object with bikes added/removed, for update purposes.
        """

        if station.current_load > station.optimal_load:
            # Take bikes
            delta_bikes = min(station.current_load - station.optimal_load,  # diff. between current and optimal load
                              self.available_space)    # truck available space to take more bikes

            # update truck and station values
            station.current_load = station.current_load - delta_bikes  # update station current_load for bakes taken out
            self.current_load = self.current_load + delta_bikes  # update truck current_load for bikes taken in
            self.available_space = self.max_load - self.current_load
            station.update_urgency()  # update station urgency value
            
        elif station.current_load < station.optimal_load:
            # Put bikes
            delta_bikes = min(station.optimal_load - station.current_load,  # diff. between current and optimal load
                              self.current_load)    # truck current load, the amount of bikes it can give away

            # update truck and station values
            station.current_load = station.current_load + delta_bikes  # update station current_load for bikes taken in
            self.current_load = self.current_load - delta_bikes  # update truck current_load for bikes taken out
            self.available_space = self.max_load - self.current_load
            station.update_urgency()  # update station urgency value
            
        else:
            delta_bikes = 0

        return delta_bikes, station


def get_station_json(url="https://api-core.bixi.com/gbfs/fr/station_information.json",
                     json_filename=None):
    """
    Get Station Information JSON from Web using requests or from a local file if the web file is not available.

    :param url: url of the station_information file from bixi's servers
    :param json_filename: filename of the json file if stored locally
    :return: returns a formatted json file containing station information
    """
    #
    try:
        r = requests.get(url)
        station_json = r.json()
        return station_json

    except requests.exceptions.ConnectionError:
        print("ConnectionError: Could not reach the Bixi API-Core server. Falling back to local resource, if available.")

        # Falling back to local json file in case connection is unavailable.
        try:
            with open(json_filename, "r") as json_file:
                station_json=json.load(fp=json_file)
                return station_json

        except:
            print("Could not load station_information.json from the script's folder. Exiting...")
            sys.exit()


def generate_station_list(_json=None):
    """
    Generates a list of Station objects from a JSON file.
    :param _json: imported json file (structured like a dictionary with the json library)
    :return: List of Station objects
    """
    assert _json is not None, "JSON must be a formatted dictionary from a JSON file"
    _station_list = []
    for _s in range(0, len(_json['data']['stations'])):
        # if json['data']['stations'][s]['capacity'] == 0:
        #    pass # if the capacity is 0, it could mean that the station is closed (our theory anyway)
        # else:
        _station_list.append(Station(name=_json['data']['stations'][_s]['name'],
                                     short_name=_json['data']['stations'][_s]['short_name'],
                                     id_=_s,
                                     bixi_id=_json['data']['stations'][_s]['station_id'],
                                     max_load=_json['data']['stations'][_s]['capacity'],
                                     latitude=_json['data']['stations'][_s]['lat'],
                                     longitude=_json['data']['stations'][_s]['lon']
                                     )
                           )
    print("The network was initialized with", len(_station_list), "stations.")
    return _station_list


def distribute_bikes(_station_list=None):
    """
    :param _station_list:
    :return:
    """
    assert _station_list is not None, "You must supply a station list"

    print("Calculating the number of bikes to supply the network with...")
    _total_bike_count = 0

    for _i in range(0, len(_station_list)):
        _total_bike_count = _total_bike_count + math.floor(_station_list[_i].max_load / 2)
    print("Total number of bikes in the network:", _total_bike_count)
    print("Initializing each station's starting bike count...")
    _initial_bikes_remaining = _total_bike_count  # initialize bikes to assign based on total_bike_count

    for _i in range(0, len(_station_list)):
        # For each station, select a random int between 0 and the station's max load: this is the
        # number of bikes to assign to the station initially
        _initial_bikes = random.randint(0, _station_list[_i].max_load)
        # if the number of bikes to assign is greater than the total amount of bikes remaining to be assigned
        # then, only assign what remains
        if _initial_bikes > _initial_bikes_remaining:
            _initial_bikes = _initial_bikes_remaining
        elif _initial_bikes <= _initial_bikes_remaining:
            pass

        _initial_bikes_remaining = _initial_bikes_remaining - _initial_bikes
        _station_list[_i].current_load = _initial_bikes
        _station_list[_i].update_urgency()

        if _initial_bikes_remaining == 0:
            break
    print(_total_bike_count, "bikes distributed across", len(_station_list), "stations.")
    return _station_list


def generate_distance_matrix(_station_list=None):
    """
    Generates the distance matrix between pairs of station.

    :param _station_list: a list of Stations objects with which to calculate distances from
    :return: a square mirror matrix, containing float values of the distance between each pair of stations
             the diagonal is initialized as 0.
    """
    assert _station_list is not None, "You must supply a station list, a list with Station objects."

    # Initiation de la matrice des distances:
    print("Initializing distance matrix...")
    _distance_matrix = np.zeros((len(_station_list), len(_station_list)))

    for _i in range(0, len(_station_list)):
        for _j in range(0, len(_station_list)):
            if _i == _j:
                _distance_matrix[_i,_j] = 0
            else:
                _distance_matrix[_i,_j] = _station_list[_i].calculate_distance(_station_list[_j])
    return _distance_matrix


def compute_urgency_metrics(_station_list=None):
    """
        Returns a tuple, containing the mean and the variance of the urgency of all stations in the _station_list
    """
    print("Computing urgency metrics on", len(_station_list), "stations")
    assert _station_list is not None, "_station_list must not be empty or None!"

    _urgency_list = []
    for _i in range(0, len(_station_list)):
        _urgency_list.append(_station_list[_i].urgency)

    _urgency_mean = sum(_urgency_list) / len(_urgency_list)
    
    # Computing variance
    _urgency_variance_sum = 0
    for _j in range(0, len(_urgency_list)):
        _urgency_variance_sum = _urgency_variance_sum + (_urgency_list[_j] - _urgency_mean) ** 2
    
    _urgency_variance = _urgency_variance_sum / len(_urgency_list)
    
    return _urgency_mean, _urgency_variance


def save_distance_matrix(_distance_matrix, filename='distance_matrix.txt'):
    """
    Saves the distance matrix to a file
    """
    # Saving distance matrix to file, just because we can.
    np.savetxt(fname=filename, X=_distance_matrix)


# SOLUTION STARTS HERE #########################################################################################
# SOLUTION STARTS HERE #########################################################################################


def optimize_solution(csvfile, log=True, double_dip=False, hardcap=1000, truck_max_load=25, initial_station_id=0,
                      plot=False, save_plot=False, initial_search_radius=0.5, max_search_radius=5, optimization_threshold=0.1):
    """
    Run the optimizer to find the best path the truck can take to distribute bikes evenly across the network.
    This is the main function to call.
    
    Arguments:
        csvfile: output file for the log. Required if log is True.

        log: default True: Controls whether to output a logfile of the optimization iterations, or not.

        double_dip: default False: Controls whether a truck can go back to a previously visited station or not.

        initial_search_radius: Default: 0.5 km, sets the starting range that a station can be to count as nearby.

        hardcap: max number of moves between two stations before stopping

        truck_max_load: maximum number of bikes that can fit aboard the truck at any one time

        max_search_radius: Default : 5km, limits the max range a next station can be before the optimizer
                                                    stops.

        initial_station_id: Default : 0, ID of the station the truck starts at

        optimization_threshold: Default: 0.1, Sets the minimum value of the network urgency variance to reach in order
                                                to stop the optimizer.
    """

    print("Initializing Solution Optimizer...")
    print("Reading JSON File...")
    # Getting data
    station_json = get_station_json("https://api-core.bixi.com/gbfs/fr/station_information.json")

    # Parsing stations
    station_list = generate_station_list(station_json)

    # distributing initial bike levels
    station_list = distribute_bikes(station_list)

    # generating distance matrix
    distance_matrix = generate_distance_matrix(station_list)

    # initializing truck (agent)
    truck = Truck(max_load=truck_max_load, current_station=station_list[initial_station_id])  # Initialisation du truck

    # Preparing CSV Log output:
    if log and csvfile is None:
        print("Warning! No CSV file output path has been provided. Log will not be generated.")
        log = False
    
    if log:
        csv_header = ["Iter_ID",
                      "Starting Station",
                      "Next Station",
                      "Next Station Urgency",
                      "Distance Driven",
                      "Nearby Stations",
                      "Nearby Stations Urgency",
                      "Next Station Load (Before)", 
                      "Next Station Availability (Before)",
                      "Action Type", 
                      "Action Bike Count", 
                      "Truck Load (Before)",
                      "Truck Load (After)",
                      "Next Station Load (After)", 
                      "Next Station Availability (After)", 
                      "Total Distance Driven", 
                      "Global Mean Urgency",
                      "Global Urgency Variance"]
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(csv_header)
        csv_current_iter_row = []

    visited_stations_lat = []
    visited_stations_lon = []
    visited_stations = []
    current_search_radius = initial_search_radius
    total_distance = 0

    optimize_iter = 0
    optimize = True

    while optimize:
        optimize_iter = optimize_iter + 1

        ###############################################################################################################
        # Getting nearby stations based on current_search_radius
        nearby_stations = []
        nearby_stations_ids = []
        for i in range(0, len(station_list)):
            if i != truck.current_station.id_:  # don't take current station into consideration
                if double_dip:  # if specified, don't select the stations already visited as potential stations
                    if distance_matrix[truck.current_station.id_, i] <= current_search_radius:
                        nearby_stations.append(station_list[i]) 
                        nearby_stations_ids.append(station_list[i].id_)
                        
                else:
                    if distance_matrix[truck.current_station.id_, i] <= current_search_radius and \
                            i not in visited_stations:
                        nearby_stations.append(station_list[i])
                        nearby_stations_ids.append(station_list[i].id_)

        ################################################################################################################
        # From the list of nearby stations (SCENARIO 1):
        # if truck load is higher than its median capacity, search for a station with a low bikes count to unload some
        if truck.current_load >= (truck.max_load / 2):
            most_urgent_station = None  # Reset the value, expects a Station object
            for j in range(0, len(nearby_stations)):
                if most_urgent_station is None:
                    most_urgent_station = nearby_stations[j]  # Initialize most urgent station as the 1st to be analyzed

                # if one of the nearby station is more urgent that the current most urgent one, select it instead
                elif nearby_stations[j].urgency > most_urgent_station.urgency:
                    most_urgent_station = nearby_stations[j]

                # if urgency levels are equal, select the one that is the closest to the current station
                elif nearby_stations[j].urgency == most_urgent_station.urgency:
                    if distance_matrix[truck.current_station.id_, j] < distance_matrix[truck.current_station.id_,
                                                                                       most_urgent_station.id_]:
                        most_urgent_station = nearby_stations[j]
                    else:
                        pass
                else:
                    pass
                
            # if we didn't find a station to move to: 
            if most_urgent_station is None:
                print("Could not find a most urgent station to move to. Is the nearby station list empty?")
                current_search_radius = current_search_radius + 0.5
                print("Increasing search radius to", current_search_radius, "km")

                if current_search_radius > max_search_radius:
                    print("Search radius over maximum search radius. Stopping.")
                    break
                else:
                    optimize = True
                    pass
            
            else: # we found a station to move to
                print("Most Urgent Station:", str(most_urgent_station.id_), most_urgent_station.name,
                      str(most_urgent_station.urgency))

                if log:
                    csv_current_iter_row = [optimize_iter,
                                            truck.current_station.id_,
                                            most_urgent_station.id_,
                                            most_urgent_station.urgency,
                                            distance_matrix[truck.current_station.id_, most_urgent_station.id_],
                                            "Not yet implemented",
                                            "Not yet implemented",
                                            most_urgent_station.current_load,
                                            most_urgent_station.max_load - most_urgent_station.current_load,
                                            "Put",
                                            0,
                                            truck.current_load,  # current load before take/put
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0]

                # update total distance value
                total_distance = total_distance + distance_matrix[truck.current_station.id_, most_urgent_station.id_]

                # move the truck to the next station
                truck.current_station = most_urgent_station

                # update visited stations lists:
                visited_stations.append(truck.current_station.id_)
                visited_stations_lat.append(truck.current_station.latitude)
                visited_stations_lon.append(truck.current_station.longitude)

                # load/unload bikes and update station network:
                iter_bikes, updated_station = truck.load_unload_bikes(truck.current_station)
                station_list[truck.current_station.id_] = updated_station
                
                # Update metrics and log
                iter_urgency_mean, iter_urgency_var = compute_urgency_metrics(station_list)
                
                # Finalize log row
                if log:
                    csv_current_iter_row[10] = iter_bikes
                    csv_current_iter_row[12] = truck.current_load
                    # 11 already updated (truck current load before update)
                    csv_current_iter_row[13] = truck.current_station.current_load
                    csv_current_iter_row[14] = truck.current_station.max_load - truck.current_station.current_load
                    csv_current_iter_row[15] = total_distance
                    csv_current_iter_row[16] = iter_urgency_mean
                    csv_current_iter_row[17] = iter_urgency_var
                    csv_writer.writerow(csv_current_iter_row)
                    csvfile.flush()
                
                print("Iteration", optimize_iter, "-- Mean Urgency:", iter_urgency_mean, "Variance:", iter_urgency_var,
                      "Station visited", truck.current_station.id_)

                # Check metrics if we stop optimizing, or continue
                if iter_urgency_var > optimization_threshold:  # continue if we haven't reached optimization threshold.
                    optimize = True
                
                if optimize_iter > hardcap:  # stop after iteration
                    optimize = False
            
        ###############################################################################################################
        # From the list of nearby stations (SCENARIO 2):
        # if truck load is lower than its median capacity, search for a station with a high bike count to load some in
        else: 
            least_urgent_station = None  # Reset the value; expects a Station object
            for j in range(0, len(nearby_stations)):
                if least_urgent_station is None:
                    least_urgent_station = nearby_stations[j]  # Initialize least urgent station as the 1st analyzed

                # if one of the nearby station is less urgent that the current least urgent one, select it instead
                elif nearby_stations[j].urgency < least_urgent_station.urgency:
                    least_urgent_station = nearby_stations[j]

                # if urgency levels are equal, select the one that is the closest to the current station
                elif nearby_stations[j].urgency == least_urgent_station.urgency:
                    if distance_matrix[truck.current_station.id_, j] < distance_matrix[truck.current_station.id_,
                                                                                       least_urgent_station.id_]:
                        least_urgent_station = nearby_stations[j]
                    else:
                        pass
                else:
                    pass
                
            if least_urgent_station is None:
                print("Could not find a least urgent station to move to. Is the nearby station list empty?")
                current_search_radius = current_search_radius + 0.5
                print("Increasing search radius to", current_search_radius, "km")
                if current_search_radius > max_search_radius:
                    print("Search radius over maximum search radius. Stopping.")
                    break
                else:
                    optimize = True
                    pass
            
            else: # we found a station to move to
                print("Least Urgent Station:", str(least_urgent_station.id_), least_urgent_station.name,
                      str(least_urgent_station.urgency))

                # Update log values
                if log:
                    csv_current_iter_row = [optimize_iter,
                                            truck.current_station.id_,
                                            least_urgent_station.id_,
                                            least_urgent_station.urgency,
                                            distance_matrix[truck.current_station.id_, least_urgent_station.id_],
                                            "Not yet implemented",
                                            "Not yet implemented",
                                            least_urgent_station.current_load,
                                            least_urgent_station.max_load - least_urgent_station.current_load,
                                            "Take",
                                            0,
                                            truck.current_load,  # current load before take/put,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0]
                # update total distance value
                total_distance = total_distance + distance_matrix[truck.current_station.id_, least_urgent_station.id_]

                # move the truck to the next station
                truck.current_station = least_urgent_station

                # update visited stations lists:
                visited_stations.append(truck.current_station.id_)
                visited_stations_lat.append(truck.current_station.latitude)
                visited_stations_lon.append(truck.current_station.longitude)

                # load/unload bikes and update station network:
                iter_bikes, updated_station = truck.load_unload_bikes(truck.current_station)
                station_list[truck.current_station.id_] = updated_station

                # Update metrics
                iter_urgency_mean, iter_urgency_var = compute_urgency_metrics(station_list)

                # Finalize log row
                if log:
                    csv_current_iter_row[10] = iter_bikes
                    # 11 already updated (truck current load before update)
                    csv_current_iter_row[12] = truck.current_load
                    csv_current_iter_row[13] = truck.current_station.current_load
                    csv_current_iter_row[14] = truck.current_station.max_load - truck.current_station.current_load
                    csv_current_iter_row[15] = total_distance
                    csv_current_iter_row[16] = iter_urgency_mean
                    csv_current_iter_row[17] = iter_urgency_var
                    csv_writer.writerow(csv_current_iter_row)
                    csvfile.flush()
                
                print("Iteration", optimize_iter, "-- Mean Urgency:", iter_urgency_mean, "Variance:", iter_urgency_var,
                      "Station visited", truck.current_station.id_)

                if iter_urgency_var < optimization_threshold:
                    print("Optimization Goal (Threshold) Reached! Optimization Complete!")
                    optimize = False

                # stop if we have done too many iterations, based on hardcap value
                if optimize_iter > hardcap:
                    print("Optimization Hardcap Reached. Stopping.")
                    optimize = False
    
    #################################################################################################################

    # End of optimization:
    print("Optimization Complete!")
    print("Visited: ", str(len(visited_stations)), "stations.")
    print("Visited Stations (Solution):")
    print(str(visited_stations))

    if save_plot:
        png_dir = os.path.join(os.getcwd(), 'png')  # register png path
        if not os.path.exists(png_dir):  # create png folder if it doesn't exist
            os.makedirs(png_dir)

    # Plotting the plot
    if plot:
        geodf = gpd.read_file("LIMADMIN.shp")
        geodf['coords'] = geodf['geometry'].apply(lambda x: x.representative_point().coords[:])
        geodf['coords'] = [coords[0] for coords in geodf['coords']]

        geodf.plot(color='white', edgecolor='black')

        for idx, row in geodf.iterrows():
            plt.annotate(s=row['NOM'], xy=row['coords'],
                         horizontalalignment='center')

        # data point colors:
        plt.rc('axes', prop_cycle=(cycler('color', ['lightcoral', 'indianred', 'brown', 'maroon', 'darkred', 'red',
                                                    'tomato', 'coral', 'lightsalmon', 'chocolate', 'saddlebrown',
                                                    'peru', 'darkorange', 'orange', 'goldenrod', 'khaki', 'olive',
                                                    'yellow', 'yellowgreen', 'darkolivegreen', 'darkseagreen',
                                                    'limegreen', 'seagreen', 'aquamarine', 'turquoise', 'lightseagreen',
                                                    'teal', 'aqua', 'deepskyblue', 'steelblue', 'lightsteelblue',
                                                    'royalblue', 'midnightblue', 'navy', 'blue', 'mediumpurple',
                                                    'indigo', 'darkviolet', 'violet', 'fuchsia', 'orchid', 'crimson'])))

        # Plotting points
        for point in range(len(visited_stations_lon)):
            plt.plot(visited_stations_lon[0:point+1], visited_stations_lat[0:point+1], '-', c='silver',
                     linestyle='dashed', linewidth=1)
            x = visited_stations_lon[point]
            y = visited_stations_lat[point]
            plot_margin = 0.001

            min_x = min(visited_stations_lon[0:point+1]) - plot_margin
            min_y = min(visited_stations_lat[0:point+1]) - plot_margin
            max_x = max(visited_stations_lon[0:point+1]) + plot_margin
            max_y = max(visited_stations_lat[0:point+1]) + plot_margin
            plt.xlim(min_x, max_x)
            plt.ylim(min_y, max_y)
            plt.plot(x, y, 'o-', )

            if save_plot:
                png_path = os.path.join(os.getcwd(), 'png', 'plot_' + str(point) + '.png')
                plt.savefig(fname=png_path)

            plt.pause(0.02)

        plt.show()

    # End of plotting section


if __name__ == '__main__':
    # Log file setup:
    start_time = str(datetime.datetime.now()).replace(":", "")  # register current time for log filename
    log_dir = os.path.join(os.getcwd(), 'log')  # register log path
    if not os.path.exists(log_dir):  # create log folder if it doesn't exist
        os.makedirs(log_dir)

    csv_filename = os.path.join(os.getcwd(), 'log', "log_" + start_time + ".csv")  # register log csv filename
    with open(csv_filename, 'w') as csvfile:  # create log csv
        # Run the optimizer with logging
        optimize_solution(csvfile, log=True, plot=True, save_plot=True, double_dip=False, initial_search_radius=0.5,
                          optimization_threshold=0.005, max_search_radius=10, truck_max_load=15, initial_station_id=0)
