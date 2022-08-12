import cv2
import numpy as np
import math
import warnings
from collections import namedtuple
import os
from random import randint
from scipy.stats import t

# Toggle printing images/graphs
print_images = False
print_graphs = False


def find_first(item, vec):
    """return the index of the first occurrence of item in vec"""
    for i in range(len(vec)):
        if item == vec[i]:
            return i
    return -1


def find_nearest(array, value):
    """Returns the index of the array element that is closest to the specified value"""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def find_distance(points, centre):
    """Returns array of distances between each points in array input 'points' and one point 'centre'"""
    subtracted_x = np.square(np.subtract(centre[0], points[:, 0]))
    subtracted_y = np.square(np.subtract(centre[1], points[:, 1]))
    distance = np.sqrt(np.add(subtracted_x, subtracted_y))

    return distance


def pre_crop_image(image, percentage=0):
    """Crops input image to a circle. This step is useful for removing stray particles from the image"""
    # Display original image
    if print_images is True:
        cv2.imshow('Original', image)
        cv2.waitKey(0)

    # Extract size
    hh, ww = image.shape[:2]
    hh2 = hh // 2
    ww2 = ww // 2

    # define circles
    radius = int((hh - (hh * (percentage / 100))) / 2)
    xc = hh // 2
    yc = ww // 2

    # Crop image to circle
    mask = np.zeros_like(image)
    mask = cv2.circle(mask, (xc, yc), radius, (255, 255, 255), -1)

    cropped_image = cv2.bitwise_and(image, mask)

    if print_images is True:
        cv2.imshow('Preprocessing Cropped Image', cropped_image)
        cv2.waitKey(0)

    return cropped_image


def binarise_image(greyscale_image, threshold_val=1):
    """Binarises grey scale image and returns black and white image"""
    # Convert image to binary black and white image
    thresh, binary_image = cv2.threshold(
        greyscale_image,
        threshold_val,
        255,
        cv2.THRESH_BINARY
    )
    binary_image = binary_image[:, :, 0]

    # Display binary image
    if print_images is True:
        cv2.imshow('Binary Image', binary_image)
        cv2.waitKey(0)

    return binary_image


def extract_free_surface_canny(binary_image, blur=False):
    """Returns the locations of the power-air boundary"""
    # Blur the image to even out noise
    if blur is True:
        new_image = cv2.GaussianBlur(binary_image, (3, 3), 0)
        # Display blurred image
        if print_images is True:
            cv2.imshow('Blurred Image', new_image)
            cv2.waitKey(0)
    else:
        new_image = binary_image

    # Canny Edge Detection
    canny_edge_image = cv2.Canny(image=new_image, threshold1=0, threshold2=10)  # Canny Edge Detection
    if print_images is True:
        cv2.imshow('Canny Edge Detection', canny_edge_image)
        cv2.waitKey(0)

    # Extract size
    hh, ww = canny_edge_image.shape[:2]

    return canny_edge_image


def extract_free_surface_ff(binary_image):
    """Returns the locations of the power-air boundary using the find first method"""
    # Find all coordinates that have values of 255 in each column
    hh, ww = binary_image.shape[:2]
    window = binary_image.transpose()
    xy = []
    for i in range(len(window)):
        col = window[i]
        j = find_first(255, col)
        xy.extend((i, j))
    # Reshape into [[x1, y1],...]
    edge_coordinates = np.array(xy).reshape((-1, 2))
    xdata = edge_coordinates[:, 0]
    ydata = np.subtract(hh, edge_coordinates[:, 1])
    edge_coordinates = np.concatenate((np.vstack(xdata), np.vstack(ydata)), axis=1)

    return edge_coordinates


def crop_points(points, image_resolution, percentage=10):
    """Returns a set of points that are within a percentage circle of the centre of the original image"""
    # Find threshold radius to crop out points outside of
    threshold_radius = (image_resolution / 2) * ((100 - percentage) / 100)

    # Find distance from centre to all points
    centre = [image_resolution / 2, image_resolution / 2]
    distance = find_distance(points, centre)

    # Threshold out points outside of crop circle
    threshold_index = np.where(distance > threshold_radius)
    points[threshold_index] = -1
    points[points == -1] = np.nan
    cropped_points = points

    return cropped_points


def process_image(image, ff_method=False):
    """Combines all image_processing methods into one method. Outputs image with the edges of the powder"""
    initial_cropped_img = pre_crop_image(image, percentage=1)
    binary_img = binarise_image(initial_cropped_img)
    if ff_method is True:
        edges = extract_free_surface_ff(binary_img)
    else:
        edge_image = extract_free_surface_canny(binary_img)

    edge_crop_image = pre_crop_image(edge_image, 10)

    # Extract size
    hh, ww = image.shape[:2]
    hh2 = hh // 2

    # Extract coordinates of powder-air interface from canny edge detection image.
    window = edge_crop_image.transpose()
    # xdata = []
    # ydata = []
    # for px, column in enumerate(window):
    #     index = np.transpose((column == 255).nonzero())
    #     if len(index) == 0:
    #         xdata.append(px)
    #         ydata.append(None)
    #     else:
    #         xdata.append(px)
    #         ydata.append(index)
    #
    # edge_coordinates = np.concatenate((np.vstack(xdata), np.vstack(ydata)), axis=1)

    edge_coordinates = np.transpose((window == 255).nonzero())
    xdata = edge_coordinates[:, 0]
    ydata = np.subtract(hh, edge_coordinates[:, 1])  # Mirror ydata to flip surface to correct orientation.
    edge_coordinates = np.array(np.concatenate((np.vstack(xdata), np.vstack(ydata)), axis=1))
    test = edge_coordinates.shape
    # grouped_ydata = np.split(edge_coordinates[:, 1], np.unique(edge_coordinates[:, 0], return_index=True)[1][1:])
    # xdata_unique = np.unique(xdata)
    # print(len(grouped_ydata))
    # print(xdata_unique)
    # interface = []
    # for px in range(hh):
    #     if px in xdata_unique:
    #         print(px)
    #         interface.extend((px, [grouped_ydata[px]]))
    #     else:
    #         interface.extend((px, None))  # If there is no xdata value at pixel px then extend array with None

    return edge_coordinates


def single_dynamic_angle_of_repose(free_surface_ordinates, diameter=500):
    """Calculates dynamic angle of repose in degrees from a set of free surface ordinates using the Granutools method"""

    # # Image size
    # h_diameter, w_diameter = image.shape[:2]
    # d_5 = w_diameter/5
    #
    # # Crop image
    # free_surface_ordinates = process_image(image)
    d_5 = diameter / 5

    # Find central point to calculate dynamic angle of repose at
    # length = len(free_surface_ordinates)
    # sum_y = np.sum(free_surface_ordinates[:, 1])
    # average_y = sum_y / length
    actual_centre = diameter/2
    centre_index = find_nearest(free_surface_ordinates[:, 0], actual_centre)
    centre = free_surface_ordinates[centre_index]

    # Find the points to use for dynamic angle of repose calculation
    ordinates_left = free_surface_ordinates[0:centre_index]
    ordinates_right = free_surface_ordinates[centre_index:len(free_surface_ordinates)]

    distance_left = centre[0] - ordinates_left[:, 0]
    distance_right = ordinates_right[:, 0] - centre[0]

    top_left_index = find_nearest(distance_left, d_5/2)
    bottom_right_index = find_nearest(distance_right, d_5/2)

    top_left = ordinates_left[top_left_index]
    bottom_right = ordinates_right[bottom_right_index]
    points = np.vstack((top_left, bottom_right))

    # Calculate angle of repose
    dynamic_angle_degrees = math.degrees(
        math.atan2((top_left[1] - bottom_right[1]),
                   (top_left[0] - bottom_right[0])))

    # Return values
    dynamic_angle = namedtuple("dynamic_angle", ["dynamic_angle_degrees", "points", "free_surface_ordinates"])

    return dynamic_angle(
        dynamic_angle_degrees,
        points,
        free_surface_ordinates
    )


def process_images(images_path, filename, n=-1):
    """Takes a directory and common filename for a series of images and extracts the free surface"""
    # Get list of images
    image_list = []
    for file in os.listdir(images_path):
        if file.startswith(filename):
            image_list.append(file)
    image_list.sort()
    if len(image_list) < 1:
        raise Exception(f'There are no images in "{images_path}" which contain "{filename}" in their filename.')

    if n == -1:
        new_image_list = image_list
    elif n < 1:
        raise Exception(f'n must be >=1 or n must be == -1 to use all images. n was found to be equal to {n}')
    else:
        new_image_list = []
        image_index = np.delete(np.linspace(0, len(image_list), n+1), 0)
        gap_size = image_index[1] - image_index[0]

        for j, image_number in enumerate(image_index):
            rand_val = randint(1, int(gap_size))  # Add some randomness to images chosen for processing
            new_image_list.append(image_list[int(image_number-rand_val)])

    all_free_surface_points = []  # Array of all free surface points for all images

    for i, image in enumerate(new_image_list):
        cv_img = cv2.imread(f'{images_path}/{image}')
        free_surface_points = process_image(cv_img)
        all_free_surface_points.append(free_surface_points)

    return all_free_surface_points


def dynamic_angle_of_repose(all_free_surface_points):
    """Calculates dynamic angle of repose for an array of free surface points taken from a series of images"""
    # Average the interface at each x coordinate
    image_resolution = 500

    averaged_interface_x = []
    averaged_interface_y = []
    for x_pixel in range(image_resolution):
        y_values = []
        for x, interface in enumerate(all_free_surface_points):
            index = np.where(interface[:, 0] == x_pixel)[0]
            if index.size == 0:
                y_values = -1
                break
            y_values.append(interface[index, 1])

        if y_values == -1:
            continue
        else:
            y_values = np.concatenate(y_values)

        mean_y = np.mean(y_values)
        averaged_interface_x.append(x_pixel)
        averaged_interface_y.append(round(mean_y, 3))

    averaged_interface = np.concatenate((
        np.vstack(averaged_interface_x),
        np.vstack(averaged_interface_y)),
        axis=1)

    # Calculate dynamic angle of repose
    dynamic_angle_data = single_dynamic_angle_of_repose(averaged_interface)
    dynamic_angle_degrees = dynamic_angle_data.dynamic_angle_degrees
    points = dynamic_angle_data.points

    # Return values
    dynamic_angle = namedtuple("dynamic_angle", ["dynamic_angle_degrees", "points", "averaged_interface"])

    return dynamic_angle(
        dynamic_angle_degrees,
        points,
        averaged_interface
    )


def cohesive_index(all_free_surface_points, average_free_surface):
    """Calculates the cohesive index from a set of interface coordinates and the average free surface"""
    # Calculate the standard deviation compared to the average interface with Granutools' equation
    standard_deviation_all = []
    conf_bounds_top = []
    conf_bounds_bottom = []
    confidence = 0.95
    for i, average_y in enumerate(average_free_surface):
        y_value = []
        x_pixel = average_free_surface[:, 0][i]
        mean = average_free_surface[:, 1][i]
        dof = 49
        for x, interface in enumerate(all_free_surface_points):
            index = np.where(interface[:, 0] == x_pixel)[0]
            y_value.append(np.mean(interface[index, 1]))  # Find mean position of interface at x_pixel for each image

        y_values = np.subtract(average_free_surface[:, 1][i], y_value)
        standard_deviation_x = np.sqrt((np.sum(np.power(y_values, 2)))/len(y_values))
        standard_deviation_all.append(standard_deviation_x)

        # Calc confidence bounds
        t_crit = np.abs(t.ppf((1 - confidence) / 2, dof))
        conf_bounds = (mean-standard_deviation_x*t_crit/np.sqrt(50), mean+standard_deviation_x*t_crit/np.sqrt(50))
        conf_bounds_top.append(conf_bounds[0])
        conf_bounds_bottom.append(conf_bounds[1])

    cohesive_index_value = np.mean(standard_deviation_all)

    return cohesive_index_value, conf_bounds_top, conf_bounds_bottom


# def confidence_bounds(average_free_surface, ):


def fit_polynomial(average_free_surface, order=3):
    """Fits a polynomial to the average free surface calculated from a series of images"""
    # Extract x and y data from array
    xdata = average_free_surface[:, 0]
    ydata = average_free_surface[:, 1]

    poly_eqn = np.polynomial.polynomial.Polynomial.fit(xdata, ydata, order)

    return poly_eqn


cv2.destroyAllWindows()
