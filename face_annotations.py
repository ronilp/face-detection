# File: face_annotations.py
# Author: Ronil Pancholia
# Date: 4/6/19
# Time: 5:26 AM

class Face_Annotations(object):

    top_left_x = 0
    top_left_y = 0
    bottom_right_x = 0
    bottom_right_y = 0

    def __init__(self, center_x, center_y, major_radius, minor_radius):
        self.center_x = center_x
        self.center_y = center_y
        self.major_radius = major_radius
        self.minor_radius = minor_radius
        return


    # convert coordinates to top-left and right-bottom
    def process_coordinates(self):
        left = max(self.center_x - self.minor_radius, 0)
        upper = max(self.center_y - self.major_radius, 0)
        bottom = self.center_y + self.major_radius
        right = self.center_x + self.minor_radius
        return left, upper, right, bottom