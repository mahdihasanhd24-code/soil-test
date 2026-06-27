import cv2
import numpy as np
import os
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

def preprocess_image(img_path):
    """
    Applies preprocessing steps described in Section 2.2 of the paper:
    1. Resizing to 250x250
    2. Color Conversion to Grayscale
    3. Noise Reduction using Median Filter
    4. Contrast Enhancement via Histogram Equalization
    Returns both the processed RGB image (for color features) and preprocessed grayscale image.
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found at {img_path}")
        
    # Read image in color (BGR)
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError(f"Failed to read image at {img_path}")
        
    # 2.2.1 Resizing: Scale to 250 x 250 pixels
    img_resized = cv2.resize(img_bgr, (250, 250))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # 2.2.2 Color Conversion: Convert RGB to grayscale
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # 2.2.3 Noise Reduction: Apply median filter (3x3 kernel size)
    gray_denoised = cv2.medianBlur(gray, 3)
    
    # 2.2.4 Contrast Enhancement: Histogram equalization
    gray_enhanced = cv2.equalizeHist(gray_denoised)
    
    return img_rgb, gray_enhanced

def extract_color_features(img_rgb, gray_enhanced):
    """
    Section 2.3.1: Color Features
    Mean, standard deviation, and histogram-based indices of RGB and grayscale.
    """
    features = []
    # Mean and standard deviation for R, G, B channels
    for i in range(3):
        channel = img_rgb[:, :, i]
        features.append(float(np.mean(channel)))
        features.append(float(np.std(channel)))
        
    # Mean and standard deviation for Grayscale
    features.append(float(np.mean(gray_enhanced)))
    features.append(float(np.std(gray_enhanced)))
    
    # Histogram-based indices (10-bin normalized histogram of grayscale)
    hist, _ = np.histogram(gray_enhanced, bins=10, range=(0, 256))
    hist = hist.astype(float) / (hist.sum() + 1e-9)
    features.extend(hist.tolist())
    
    return features

def extract_shape_features(gray_enhanced):
    """
    Section 2.3.2: Shape Features
    Area, perimeter, aspect ratio, and compactness from contours.
    """
    # Simple binarization to extract shapes/particles
    _, thresh = cv2.threshold(gray_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    area = 0.0
    perimeter = 0.0
    aspect_ratio = 1.0
    compactness = 0.0
    
    if contours:
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(largest_contour))
        perimeter = float(cv2.arcLength(largest_contour, True))
        
        # Bounding rect for aspect ratio
        x, y, w, h = cv2.boundingRect(largest_contour)
        if h > 0:
            aspect_ratio = float(w) / h
            
        # Compactness = (perimeter^2) / (4 * pi * area)
        if area > 0:
            compactness = (perimeter ** 2) / (4 * np.pi * area)
            
    return [area, perimeter, aspect_ratio, compactness]

def extract_texture_glcm_features(gray_enhanced):
    """
    Section 2.3.3: Texture Features (GLCM)
    Contrast, correlation, energy, and homogeneity.
    """
    # Compute GLCM with 1 & 2 pixel distances and 4 directions
    glcm = graycomatrix(gray_enhanced, distances=[1, 2], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                        levels=256, symmetric=True, normed=True)
    
    contrast = float(np.mean(graycoprops(glcm, 'contrast')))
    correlation = float(np.mean(graycoprops(glcm, 'correlation')))
    energy = float(np.mean(graycoprops(glcm, 'energy')))
    homogeneity = float(np.mean(graycoprops(glcm, 'homogeneity')))
    
    return [contrast, correlation, energy, homogeneity]

def extract_texture_lbp_features(gray_enhanced):
    """
    Section 2.3.4: Local Binary Pattern (LBP) Features
    LBP histogram representing small-scale texture changes.
    """
    # LBP parameters: radius=1, points=8
    radius = 1
    n_points = 8
    lbp = local_binary_pattern(gray_enhanced, n_points, radius, method='uniform')
    
    # Compute normalized histogram of LBP (uniform LBP with P=8 has 10 bins)
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins))
    hist = hist.astype(float) / (hist.sum() + 1e-9)
    
    return hist.tolist()

def extract_surface_features(gray_enhanced):
    """
    Section 2.3.5: Surface Features
    Roughness, gradient-based indices, edge density, and local variation.
    """
    # Gradient magnitude using Sobel filters
    sobelx = cv2.Sobel(gray_enhanced, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_enhanced, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    
    mean_grad = float(np.mean(grad_mag))
    std_grad = float(np.std(grad_mag))
    
    # Edge density using Canny edge detector
    edges = cv2.Canny(gray_enhanced, 50, 150)
    edge_density = float(np.sum(edges == 255)) / edges.size
    
    # Local variation (standard deviation of the image)
    local_variation = float(np.std(gray_enhanced))
    
    return [mean_grad, std_grad, edge_density, local_variation]

def extract_statistical_features(gray_enhanced):
    """
    Section 2.3.6: Statistical Features
    Mean, variance, standard deviation, skewness, and kurtosis.
    """
    mean = float(np.mean(gray_enhanced))
    variance = float(np.var(gray_enhanced))
    std_dev = float(np.std(gray_enhanced))
    
    # Skewness and kurtosis
    flat_gray = gray_enhanced.flatten().astype(float)
    mean_diff = flat_gray - mean
    
    skewness = 0.0
    kurtosis = 0.0
    
    if std_dev > 0:
        skewness = float(np.mean(mean_diff ** 3) / (std_dev ** 3))
        kurtosis = float(np.mean(mean_diff ** 4) / (std_dev ** 4)) - 3.0 # Excess kurtosis
        
    return [mean, variance, std_dev, skewness, kurtosis]

def extract_all_features_dict(img_path):
    """
    Extracts all feature groups and returns them in a dictionary.
    """
    img_rgb, gray_enhanced = preprocess_image(img_path)
    
    color_feats = extract_color_features(img_rgb, gray_enhanced)
    shape_feats = extract_shape_features(gray_enhanced)
    glcm_feats = extract_texture_glcm_features(gray_enhanced)
    lbp_feats = extract_texture_lbp_features(gray_enhanced)
    surface_feats = extract_surface_features(gray_enhanced)
    stat_feats = extract_statistical_features(gray_enhanced)
    
    return {
        "color": color_feats,
        "shape": shape_feats,
        "texture_glcm": glcm_feats,
        "texture_lbp": lbp_feats,
        "surface": surface_feats,
        "statistical": stat_feats
    }

def get_feature_vector_by_setting(feats_dict, setting_num=12):
    """
    Combines feature groups based on the 12 settings in Table 2 of the paper:
    1. only Color: color
    2. Texture_GLCM: texture_glcm
    3. Texture_LBP: texture_lbp
    4. Texture_GLCMLBP: texture_glcm + texture_lbp
    5. Color Texture: color + texture_glcm + texture_lbp
    6. Color & Shape: color + shape
    7. Shape Texture: shape + texture_glcm + texture_lbp
    8. Shape Statistical: shape + statistical
    9. Surface: surface
    10. Shape: shape
    11. Statistical: statistical
    12. All_Features: color + shape + texture_glcm + texture_lbp + surface + statistical
    """
    if setting_num == 1:
        return feats_dict["color"]
    elif setting_num == 2:
        return feats_dict["texture_glcm"]
    elif setting_num == 3:
        return feats_dict["texture_lbp"]
    elif setting_num == 4:
        return feats_dict["texture_glcm"] + feats_dict["texture_lbp"]
    elif setting_num == 5:
        return feats_dict["color"] + feats_dict["texture_glcm"] + feats_dict["texture_lbp"]
    elif setting_num == 6:
        return feats_dict["color"] + feats_dict["shape"]
    elif setting_num == 7:
        return feats_dict["shape"] + feats_dict["texture_glcm"] + feats_dict["texture_lbp"]
    elif setting_num == 8:
        return feats_dict["shape"] + feats_dict["statistical"]
    elif setting_num == 9:
        return feats_dict["surface"]
    elif setting_num == 10:
        return feats_dict["shape"]
    elif setting_num == 11:
        return feats_dict["statistical"]
    elif setting_num == 12:
        return (feats_dict["color"] + feats_dict["shape"] + feats_dict["texture_glcm"] + 
                feats_dict["texture_lbp"] + feats_dict["surface"] + feats_dict["statistical"])
    else:
        # Default fallback to All Features
        return (feats_dict["color"] + feats_dict["shape"] + feats_dict["texture_glcm"] + 
                feats_dict["texture_lbp"] + feats_dict["surface"] + feats_dict["statistical"])
