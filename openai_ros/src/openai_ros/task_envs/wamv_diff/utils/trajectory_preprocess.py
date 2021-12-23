import pandas as pd
from geometry_msgs.msg import Point
import numpy as np
import math

def read_trajectory_to_points(file_path, dec=1):
    '''
    Read trajectory from file and return a list of points
    :param file_path: trajectory file path
    :param dec: precision digits
    '''
    df = pd.read_csv(file_path)
    trajectory = [Point(x=round(row.x, dec), y=round(row.y, dec)) for row in df.itertuples()]
    return trajectory

def angle_diff(source, target):
    '''
    Calculate the angle difference between two angles
    :param source: source angle
    :param target: target angle
    '''
    diff = target - source
    if diff > math.pi:
        diff -= 2 * math.pi
    elif diff < -math.pi:
        diff += 2 * math.pi
    return diff


if __name__ == "__main__":
    file_path = "/home/data/code/wxc/noetic_ws/src/openai_ros/openai_ros/src/openai_ros/task_envs/wamv_diff/data/trajectory/route_0.csv"
    trajectory = read_trajectory_to_points(file_path)
    pass