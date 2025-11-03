import numpy as np
from PIL import Image
import random
from scipy.ndimage import zoom
import json
from sklearn.cluster import DBSCAN
import os
def load_image(path):
    return np.array(Image.open(path).convert("RGBA"))
import cv2
from sklearn.cluster import KMeans
import numpy as np
from collections import Counter
# from mmseg.utils import (collect_env, get_device, get_root_logger, setup_multi_processes)

def categorize_colors(colors, n_clusters=5):
    # Convert list of colors to a NumPy array
    color_array = np.array(colors)

    
    # Use KMeans clustering to group similar colors
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(color_array)
    
    # Find the most common cluster

    labels, counts = np.unique(kmeans.labels_, return_counts=True)

    most_common_cluster = labels[np.argmax(counts)]
    
    # Get the elements of the biggest group
    biggest_group = color_array[kmeans.labels_ == most_common_cluster]
    
    return biggest_group.tolist()



def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_threshold(image):
    _, binary_image = cv2.threshold(image, 125, 255, cv2.THRESH_BINARY)
    return binary_image

def calculate_similarity(image1, image2):
    # Ensure the images are the same size
    if image1.shape != image2.shape:
        raise ValueError("Images must be the same size for comparison")

    # Calculate the percentage of matching pixels
    matching_pixels = np.sum(image1 == image2)
    total_pixels = image1.size
    similarity = matching_pixels / total_pixels
    return similarity

def calculate_combined_similarity(image1, image2):
    """Return the simialri [0,1] between image 1 and image 2, which calculated by background and text"""
    # Convert images to grayscale
    gray_image1 = convert_to_grayscale(image1)
    gray_image2 = convert_to_grayscale(image2)

    # Apply threshold to convert to binary images
    binary_image1 = apply_threshold(gray_image1)
    binary_image2 = apply_threshold(gray_image2)

    # Calculate background similarity
    background_similarity = calculate_similarity(image1 * (binary_image1[:,:, None]), image2 * (binary_image2[:, :, None]))

    # Calculate text similarity
    text_similarity = calculate_similarity(binary_image1, binary_image2)

    # Combine similarity scores (you can adjust the weights as needed)
    combined_similarity = (0.8 * background_similarity + 0.2 * text_similarity)
    return combined_similarity


def extract_tampered_content(image, boxes):
    x, y, w ,h = boxes
    return image[y:y+h, x:x+w]

    # mask = np.zeros_like(image)
    # mask[y:y+h, x:x+w] = 1
    # return image * mask

def most_common_color(image, mode="bg", k=10):
    # Reshape the image to a 2D array of pixels
    pixels = image.reshape(-1, image.shape[-1])
    # Convert to list of tuples
    pixels = [tuple(pixel)[:-1] for pixel in pixels if tuple(pixel) != (0,0,0,0)]
    # Find the most common color
    
    counter = Counter(pixels)       
    colors = [[r,g,b] for (r,g,b) in list(counter.keys())]
    most_commons = categorize_colors(colors, n_clusters=k)
    most_commons = [[r,g,b, 255] for [r,g,b] in most_commons]
    return [np.array(most_common) for most_common in most_commons]

def replace_color(image, old_color, new_color):
    # Create a mask for the old color
    mask = np.all(image == old_color, axis=-1)
    # Replace the old color with the new color
    image[mask] = new_color
    return image

# Assuming image1 and image2 are already defined and preprocessed as in your code
def repaint(image1, image2):
    # Convert images to grayscale
    gray_image1 = convert_to_grayscale(image1)
    gray_image2 = convert_to_grayscale(image2)

    # Apply threshold to convert to binary images
    binary_image1 = apply_threshold(gray_image1)
    binary_image2 = apply_threshold(gray_image2)

    background_image_1 = image1 * (binary_image1[:,:,None] != 0)
    background_image_2 = image2 * (binary_image2[:,:,None] != 0)

    # Find the most common color in background_image_1
    final_image = image2+0 
    try:
        most_colors_background_1 = most_common_color(background_image_1, k=2)
        most_colors_background_2 = most_common_color(background_image_2, k=2)
        for orgcolor in most_colors_background_2:
            newcolor = random.choice(most_colors_background_1)
            final_image = replace_color(final_image, orgcolor, newcolor)
    
    except Exception as e:
        # print("------>error in replace color as", e, e.__context__)
        final_image = image2    
    # Image.fromarray(final_image).save("f.png")
    # Image.fromarray(image2).save("o.png")
    return final_image 

def tampered_augmentation(file_name, paths, save_paths, tampered_parts, k=0.5):
    img_path, ano_path = paths
    img_spath, ann_spath = save_paths
    org_annot = load_image(ano_path)
    org_image = load_image(img_path)

    tmp_image = org_image+0
    tampered_position = []
    
    k = int(len(tampered_parts) * abs(k)) if abs(k) < 1 else len(tampered_parts)//2
    while k>0:
        k = k-1
        # Select random tampered parts
        if len(tampered_parts) < 2: break

        random.shuffle(tampered_parts)

        [x1, y1, w1, h1] = tampered_parts.pop(1)
        
        if w1*h1 < 100: continue
        options                 = tampered_parts+tampered_position
        options                 = random.choices(options, k= max(1, len(options) // 2))
        target_part_1           = extract_tampered_content(tmp_image, [x1, y1, w1, h1])
        attack_options          = [zoom(extract_tampered_content(tmp_image, [x2, y2, w2, h2]), (h1 / h2, w1 / w2, 1), order=0)
                                         for  x2, y2, w2, h2 in options]
        attack_similarity       = np.array([calculate_combined_similarity(target_part_1, _p) for _p in attack_options])
        max_index               = np.argmax(attack_similarity)
        
        attack_part_resized = attack_options[max_index]
        #refine background of attack_part_resized
        # try:
        #     attack_part_resized =  repaint(target_part_1, attack_part_resized)
        # except: 
        #     attack_part_resized=attack_options[max_index]

        # Replace content of part1 with resized part2
        tampered_position += [[x1, y1, w1, h1]]

        tmp_image[y1:y1+h1, x1:x1+w1] = attack_part_resized #0 #resized_part2
        org_annot[y1:y1+h1, x1:x1+w1, :-1] = 1

    def __save__(image, path, file_name, type="RGB"):
        tampered_image_pil = Image.fromarray(image).convert(type)
        tampered_image_pil.save(os.path.join(path, file_name))
        return 
    
    __save__(tmp_image, path=img_spath, file_name=file_name)
    __save__(org_annot, path=ann_spath, file_name=file_name.replace(".jpg", ".png"), type="L")
    # __save__(tmp_image, path="", file_name=file_name)
    # __save__(np.clip(org_annot, 0, 255) * 255, path="", file_name=file_name+"_ann.png")
    # return tampered_image_pil, save_path


# def tampered_augmentation_2(org_image:np.array, org_annot:np.array, tampered_parts:list, k:int=0.5):
#     # img_path, ano_path = paths
#     # img_spath, ann_spath = save_paths
#     # org_annot = load_image(ano_path)
#     # org_image = load_image(img_path)
#     tmp_image = org_image+0
#     new_annot = org_annot + 0
#     tampered_position = []
#     k = int(len(tampered_parts) * abs(k)) if abs(k) < 1 else len(tampered_parts)//2
#     while k>0:
#         k = k-1
#         # Select random tampered parts
#         if len(tampered_parts) < 2: break
#         random.shuffle(tampered_parts)
#         [x1, y1, w1, h1] = tampered_parts.pop(1)
#         if w1*h1 < 100: continue
#         options                 = tampered_parts+tampered_position
#         options                 = random.choices(options, k= max(1, len(options) // 2))
#         target_part_1           = extract_tampered_content(tmp_image, [x1, y1, w1, h1])
#         attack_options          = [zoom(extract_tampered_content(tmp_image, [x2, y2, w2, h2]), (h1 / h2, w1 / w2, 1), order=0)
#                                          for  x2, y2, w2, h2 in options]
#         attack_similarity       = np.array([calculate_combined_similarity(target_part_1, _p) for _p in attack_options])
#         max_index               = np.argmax(attack_similarity)
        
#         attack_part_resized = attack_options[max_index]
#         tampered_position += [[x1, y1, w1, h1]]

#         tmp_image[y1:y1+h1, x1:x1+w1] = attack_part_resized #0 #resized_part2
#         new_annot[y1:y1+h1, x1:x1+w1] = 1
#     return tmp_image, new_annot


def find_boxes(matrix):
    matrix = np.array(matrix, dtype=np.uint8)
    contours, _ = cv2.findContours(matrix, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append([x,y, w,h])

    return boxes

def tampered_augmentation_2(org_image:np.array, org_annot:np.array, tampered_parts:list, k:int=0.5):

    tmp_image = org_image+0
    new_annot = org_annot + 0
    org_tampered_parts = find_boxes(org_annot)

    if len(tampered_parts) == 0: tampered_parts = org_tampered_parts.copy() 
    
    k = max(len(tampered_parts) * abs(k) if abs(k) < 1 else len(tampered_parts)//2, 1)
    

    while k>0:
        random.shuffle(tampered_parts)
        [x1, y1, w1, h1] = tampered_parts.pop(1)
        # if w1*h1 < 100: continue
        k = k-1
        target_part_1           = extract_tampered_content(tmp_image, [x1, y1, w1, h1])
        attack_options          = [zoom(extract_tampered_content(org_image, [x2, y2, w2, h2]), 
                                                                (h1 / h2, w1 / w2, 1), order=0)
                                        for  x2, y2, w2, h2 in org_tampered_parts]
        attack_similarity       = np.array([calculate_combined_similarity(target_part_1, _p) for _p in attack_options])
        max_index               = np.argmax(attack_similarity)

        attack_part_resized = attack_options[max_index]
        tmp_image[y1:y1+h1, x1:x1+w1] = attack_part_resized #0 #resized_part2
        new_annot[y1:y1+h1, x1:x1+w1] = 1
    return tmp_image, new_annot


if __name__ == "__main__":
    path='/home/k64t/Tampereddoc/zda_reimplementation/tampereddocgen/output2/annotation/0000000000_c0.json'
    # tampered_augmentation_test(path)