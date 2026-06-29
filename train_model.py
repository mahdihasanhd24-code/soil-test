import os
import json
import time
import pickle
import numpy as np
import cv2
from PIL import Image
from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_curve, auc

def extract_features(img, img_size=(250, 250)):
    # Convert PIL Image to RGB numpy array
    img_rgb = np.array(img.convert('RGB'))
    
    # 2.2.1 Resizing
    img_resized = cv2.resize(img_rgb, img_size)
    
    # 2.2.2 Grayscale conversion
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    
    # 2.2.3 Noise reduction using median filtering
    median = cv2.medianBlur(gray, 3)
    
    # 2.2.4 Contrast enhancement using histogram equalization
    equalized = cv2.equalizeHist(median)
    
    # --- Feature Extraction ---
    
    # 1. Color Features (from RGB and Grayscale)
    # RGB Mean and Std (6 features)
    mean_rgb = img_resized.mean(axis=(0, 1)) / 255.0
    std_rgb = img_resized.std(axis=(0, 1)) / 255.0
    # Grayscale Mean and Std (2 features)
    mean_gray = np.array([gray.mean() / 255.0])
    std_gray = np.array([gray.std() / 255.0])
    # Histograms
    hist_r, _ = np.histogram(img_resized[:,:,0], bins=8, range=(0, 256), density=True)
    hist_g, _ = np.histogram(img_resized[:,:,1], bins=8, range=(0, 256), density=True)
    hist_b, _ = np.histogram(img_resized[:,:,2], bins=8, range=(0, 256), density=True)
    hist_gray, _ = np.histogram(gray, bins=8, range=(0, 256), density=True)
    
    color_feats = np.concatenate([mean_rgb, std_rgb, mean_gray, std_gray, hist_r, hist_g, hist_b, hist_gray])
    
    # 2. Shape Features (from Thresholded Grayscale Image)
    _, thresh = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    num_contours = len(contours)
    if num_contours > 0:
        areas = [cv2.contourArea(c) for c in contours]
        perimeters = [cv2.arcLength(c, True) for c in contours]
        avg_area = np.mean(areas)
        avg_perimeter = np.mean(perimeters)
        
        # Calculate aspect ratio and compactness for largest contour
        largest_cnt = contours[np.argmax(areas)]
        x, y, w, h = cv2.boundingRect(largest_cnt)
        aspect_ratio = float(w) / h if h > 0 else 1.0
        
        perimeter_l = cv2.arcLength(largest_cnt, True)
        area_l = cv2.contourArea(largest_cnt)
        compactness = (4 * np.pi * area_l) / (perimeter_l ** 2) if perimeter_l > 0 else 1.0
    else:
        avg_area = 0.0
        avg_perimeter = 0.0
        aspect_ratio = 1.0
        compactness = 1.0
        
    shape_feats = np.array([float(num_contours), avg_area, avg_perimeter, aspect_ratio, compactness])
    
    # 3. Texture Features (GLCM)
    glcm = graycomatrix(equalized, distances=[1, 2], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').flatten()
    correlation = graycoprops(glcm, 'correlation').flatten()
    energy = graycoprops(glcm, 'energy').flatten()
    homogeneity = graycoprops(glcm, 'homogeneity').flatten()
    texture_feats = np.concatenate([contrast, correlation, energy, homogeneity])
    
    # 4. LBP Features
    lbp = local_binary_pattern(equalized, P=8, R=1, method='uniform')
    hist_lbp, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
    
    # 5. Surface Features
    sobelx = cv2.Sobel(equalized, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(equalized, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    mean_grad = np.mean(grad_mag)
    std_grad = np.std(grad_mag)
    
    edges = cv2.Canny(equalized, 100, 200)
    edge_density = np.sum(edges > 0) / (img_size[0] * img_size[1])
    
    local_variation = np.std(equalized)
    surface_feats = np.array([mean_grad, std_grad, edge_density, local_variation])
    
    # 6. Statistical Features
    mean_val = np.mean(equalized)
    var_val = np.var(equalized)
    std_val = np.std(equalized)
    skew_val = skew(equalized.flatten())
    kurt_val = kurtosis(equalized.flatten())
    stat_feats = np.array([mean_val, var_val, std_val, skew_val, kurt_val])
    
    # Concatenate all features (96 features)
    features = np.concatenate([color_feats, shape_feats, texture_feats, hist_lbp, surface_feats, stat_feats])
    return np.nan_to_num(features)

def load_split_dataset(split_dir, split_name, selected_classes, img_size=(250, 250)):
    X, y = [], []
    directory = os.path.join(split_dir, split_name)
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Split directory {directory} does not exist.")
        
    for label_idx, cls in enumerate(selected_classes):
        cls_dir = os.path.join(directory, cls)
        if not os.path.exists(cls_dir):
            continue
        for f in os.listdir(cls_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                path = os.path.join(cls_dir, f)
                try:
                    with Image.open(path) as img:
                        feats = extract_features(img, img_size)
                        X.append(feats)
                        y.append(label_idx)
                except Exception:
                    pass
    return np.array(X), np.array(y)

def downsample_curve(fpr, tpr, n_points=50):
    if len(fpr) <= n_points:
        return fpr.tolist(), tpr.tolist()
    indices = np.linspace(0, len(fpr) - 1, n_points, dtype=int)
    return fpr[indices].tolist(), tpr[indices].tolist()

def save_status(status_filepath, status="training", progress=0, current_epoch=0, total_epochs=3, metrics=None, logs=None, history=None):
    status_data = {
        "status": status,
        "progress": progress,
        "current_epoch": current_epoch,
        "total_epochs": total_epochs,
        "metrics": metrics or {"loss": 0.0, "accuracy": 0.0, "val_loss": 0.0, "val_accuracy": 0.0},
        "history": history or [],
        "logs": logs or []
    }
    with open(status_filepath, 'w') as f:
        json.dump(status_data, f, indent=2)

def plot_comparison(svm_overall, knn_overall, dt_overall, output_path):
    try:
        import matplotlib.pyplot as plt
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        svm_vals = [svm_overall['accuracy'], svm_overall['precision'], svm_overall['recall'], svm_overall['f1_score']]
        knn_vals = [knn_overall['accuracy'], knn_overall['precision'], knn_overall['recall'], knn_overall['f1_score']]
        dt_vals = [dt_overall['accuracy'], dt_overall['precision'], dt_overall['recall'], dt_overall['f1_score']]
        
        x = np.arange(len(metrics))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(8, 5))
        rects1 = ax.bar(x - width, svm_vals, width, label='SVM', color='#10b981')
        rects2 = ax.bar(x, knn_vals, width, label='KNN', color='#3b82f6')
        rects3 = ax.bar(x + width, dt_vals, width, label='Decision Tree', color='#d97706')
        
        ax.set_ylabel('Score (0-1)')
        ax.set_title('Soil Classifier Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1.1)
        ax.legend()
        
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height*100:.1f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)
                
        autolabel(rects1)
        autolabel(rects2)
        autolabel(rects3)
        
        fig.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error plotting comparison: {e}")

def plot_confusion_matrix_img(cm, classes, title, output_path):
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        short_classes = [c.replace(' Soil', '') for c in classes]
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=short_classes,
               yticklabels=short_classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        fmt = 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error plotting confusion matrix: {e}")

def plot_roc_curves_img(roc_auc_data, title, output_path):
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        
        for cls, data in roc_auc_data.items():
            ax.plot(data['fpr'], data['tpr'], label=f'{cls} (AUC = {data["auc"]:.3f})')
            
        ax.plot([0, 1], [0, 1], 'k--', label='Random Guess')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error plotting ROC curves: {e}")

def compute_model_metrics(model, X_test, y_test, selected_classes):
    y_pred = model.predict(X_test)
    y_pred_probs = model.predict_proba(X_test)
    
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
    recall = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
    
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(selected_classes))))
    cm_data = {
        "classes": selected_classes,
        "matrix": cm.tolist()
    }
    
    roc_auc_data = {}
    for i, cls in enumerate(selected_classes):
        y_true_binary = (y_test == i).astype(int)
        probs_cls = y_pred_probs[:, i]
        
        if len(np.unique(y_true_binary)) > 1:
            fpr, tpr, _ = roc_curve(y_true_binary, probs_cls)
            roc_auc = float(auc(fpr, tpr))
            fpr_ds, tpr_ds = downsample_curve(fpr, tpr, n_points=50)
        else:
            fpr_ds, tpr_ds = [0.0, 1.0], [0.0, 1.0]
            roc_auc = 0.5
        
        roc_auc_data[cls] = {
            "fpr": fpr_ds,
            "tpr": tpr_ds,
            "auc": roc_auc
        }
        
    return {
        "overall": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        },
        "confusion_matrix": cm_data,
        "roc_auc": roc_auc_data
    }

def train_and_evaluate(selected_classes, epochs=10, batch_size=32, learning_rate=0.001,
                       split_dir=None,
                       models_dir=None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if split_dir is None:
        split_dir = os.path.join(base_dir, "DataSet")
    if models_dir is None:
        models_dir = os.path.join(base_dir, "models")
    # If split_dir does not contain train/val/test subfolders, create them by splitting the dataset
    import random, shutil
    required_splits = ['train', 'val', 'test']
    if not all(os.path.isdir(os.path.join(split_dir, s)) for s in required_splits):
        # Expect original dataset structure: class subfolders directly under split_dir
        # Create a new folder 'dataset_split' to store splits
        split_root = os.path.join(os.path.dirname(split_dir), 'dataset_split')
        os.makedirs(split_root, exist_ok=True)
        for split in required_splits:
            os.makedirs(os.path.join(split_root, split), exist_ok=True)
        # Gather class directories
        class_dirs = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
        for cls in class_dirs:
            src_cls_dir = os.path.join(split_dir, cls)
            files = [f for f in os.listdir(src_cls_dir) if os.path.isfile(os.path.join(src_cls_dir, f))]
            random.shuffle(files)
            n = len(files)
            train_end = int(0.7 * n)
            val_end = int(0.85 * n)
            splits = {
                'train': files[:train_end],
                'val': files[train_end:val_end],
                'test': files[val_end:]
            }
            for split_name, split_files in splits.items():
                dest_dir = os.path.join(split_root, split_name, cls)
                os.makedirs(dest_dir, exist_ok=True)
                for f in split_files:
                    shutil.copy2(os.path.join(src_cls_dir, f), dest_dir)
        # Update split_dir to point to the newly created splits
        split_dir = split_root

    
    os.makedirs(models_dir, exist_ok=True)
    status_filepath = os.path.join(models_dir, "training_status.json")
    metrics_filepath = os.path.join(models_dir, "soil_metrics.json")
    
    logs_list = ["[INFO] Initializing ML training pipeline (SVM, KNN, Decision Tree)..."]
    save_status(status_filepath, status="training", progress=5, current_epoch=0, total_epochs=3, logs=logs_list)
    
    try:
        # 1. Load Datasets and extract features
        logs_list.append("[INFO] Loading split dataset directories and extracting features (32x32 pixels, color mean/std, histograms)...")
        save_status(status_filepath, status="training", progress=10, current_epoch=0, total_epochs=3, logs=logs_list)
        
        X_train, y_train = load_split_dataset(split_dir, "train", selected_classes)
        X_val, y_val = load_split_dataset(split_dir, "val", selected_classes)
        X_test, y_test = load_split_dataset(split_dir, "test", selected_classes)
        
        logs_list.append(f"[INFO] Loaded {len(X_train)} train, {len(X_val)} validation, {len(X_test)} test samples.")
        save_status(status_filepath, status="training", progress=25, current_epoch=0, total_epochs=3, logs=logs_list)
        
        model_metrics = {}
        history = []
        
        # 2. Train Support Vector Machine (SVM)
        logs_list.append("[EPOCH] Starting Model 1/3: Support Vector Machine (SVM)...")
        logs_list.append("[INFO] Training SVM with RBF kernel and C=10.0...")
        save_status(status_filepath, status="training", progress=30, current_epoch=1, total_epochs=3, logs=logs_list)
        
        t0 = time.time()
        svm_model = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0, probability=True, random_state=42))
        svm_model.fit(X_train, y_train)
        t_svm = time.time() - t0
        
        svm_results = compute_model_metrics(svm_model, X_test, y_test, selected_classes)
        model_metrics["svm"] = svm_results
        
        with open(os.path.join(models_dir, "svm_model.pkl"), 'wb') as f:
            pickle.dump(svm_model, f)
            
        plot_confusion_matrix_img(np.array(svm_results['confusion_matrix']['matrix']), selected_classes, 'SVM Confusion Matrix', os.path.join(models_dir, 'confusion_matrix_svm.png'))
        plot_roc_curves_img(svm_results['roc_auc'], 'SVM ROC Curves', os.path.join(models_dir, 'roc_curve_svm.png'))
        
        logs_list.append(f"[METRIC] SVM complete in {t_svm:.2f}s | Test Accuracy: {svm_results['overall']['accuracy']*100:.2f}%")
        
        history.append({
            "epoch": 1,
            "loss": float(1.0 - svm_results['overall']['accuracy']),
            "accuracy": float(svm_results['overall']['accuracy']),
            "val_loss": 0.0,
            "val_accuracy": float(svm_results['overall']['accuracy'])
        })
        save_status(status_filepath, status="training", progress=55, current_epoch=1, total_epochs=3, 
                    metrics={"loss": 1.0 - svm_results['overall']['accuracy'], "accuracy": svm_results['overall']['accuracy'], "val_loss": 0.0, "val_accuracy": svm_results['overall']['accuracy']},
                    history=history, logs=logs_list)
        
        # 3. Train K-Nearest Neighbors (KNN)
        logs_list.append("[EPOCH] Starting Model 2/3: K-Nearest Neighbors (KNN)...")
        logs_list.append("[INFO] Training KNN with n_neighbors=5...")
        save_status(status_filepath, status="training", progress=60, current_epoch=2, total_epochs=3, logs=logs_list, history=history)
        
        t0 = time.time()
        knn_model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
        knn_model.fit(X_train, y_train)
        t_knn = time.time() - t0
        
        knn_results = compute_model_metrics(knn_model, X_test, y_test, selected_classes)
        model_metrics["knn"] = knn_results
        
        with open(os.path.join(models_dir, "knn_model.pkl"), 'wb') as f:
            pickle.dump(knn_model, f)
            
        plot_confusion_matrix_img(np.array(knn_results['confusion_matrix']['matrix']), selected_classes, 'KNN Confusion Matrix', os.path.join(models_dir, 'confusion_matrix_knn.png'))
        plot_roc_curves_img(knn_results['roc_auc'], 'KNN ROC Curves', os.path.join(models_dir, 'roc_curve_knn.png'))
        
        logs_list.append(f"[METRIC] KNN complete in {t_knn:.2f}s | Test Accuracy: {knn_results['overall']['accuracy']*100:.2f}%")
        
        history.append({
            "epoch": 2,
            "loss": float(1.0 - knn_results['overall']['accuracy']),
            "accuracy": float(knn_results['overall']['accuracy']),
            "val_loss": 0.0,
            "val_accuracy": float(knn_results['overall']['accuracy'])
        })
        save_status(status_filepath, status="training", progress=80, current_epoch=2, total_epochs=3, 
                    metrics={"loss": 1.0 - knn_results['overall']['accuracy'], "accuracy": knn_results['overall']['accuracy'], "val_loss": 0.0, "val_accuracy": knn_results['overall']['accuracy']},
                    history=history, logs=logs_list)
        
        # 4. Train Decision Tree (DT)
        logs_list.append("[EPOCH] Starting Model 3/3: Decision Tree (DT)...")
        logs_list.append("[INFO] Training Decision Tree with max_depth=12...")
        save_status(status_filepath, status="training", progress=85, current_epoch=3, total_epochs=3, logs=logs_list, history=history)
        
        t0 = time.time()
        dt_model = DecisionTreeClassifier(max_depth=12, min_samples_leaf=5, random_state=42)
        dt_model.fit(X_train, y_train)
        t_dt = time.time() - t0
        
        dt_results = compute_model_metrics(dt_model, X_test, y_test, selected_classes)
        model_metrics["dt"] = dt_results
        
        with open(os.path.join(models_dir, "dt_model.pkl"), 'wb') as f:
            pickle.dump(dt_model, f)
            
        plot_confusion_matrix_img(np.array(dt_results['confusion_matrix']['matrix']), selected_classes, 'Decision Tree Confusion Matrix', os.path.join(models_dir, 'confusion_matrix_dt.png'))
        plot_roc_curves_img(dt_results['roc_auc'], 'Decision Tree ROC Curves', os.path.join(models_dir, 'roc_curve_dt.png'))
        
        logs_list.append(f"[METRIC] Decision Tree complete in {t_dt:.2f}s | Test Accuracy: {dt_results['overall']['accuracy']*100:.2f}%")
        
        history.append({
            "epoch": 3,
            "loss": float(1.0 - dt_results['overall']['accuracy']),
            "accuracy": float(dt_results['overall']['accuracy']),
            "val_loss": 0.0,
            "val_accuracy": float(dt_results['overall']['accuracy'])
        })
        
        # 5. Plot Comparison and Save
        logs_list.append("[INFO] Saving comparison chart to models/performance_comparison.png...")
        plot_comparison(svm_results['overall'], knn_results['overall'], dt_results['overall'], os.path.join(models_dir, 'performance_comparison.png'))
        
        # Determine best model
        best_name = "svm"
        best_f1 = svm_results['overall']['f1_score']
        if knn_results['overall']['f1_score'] > best_f1:
            best_name = "knn"
            best_f1 = knn_results['overall']['f1_score']
        if dt_results['overall']['f1_score'] > best_f1:
            best_name = "dt"
            best_f1 = dt_results['overall']['f1_score']
            
        best_results = model_metrics[best_name]
        
        metrics_summary = {
            "models": model_metrics,
            "comparison": {
                "classes": ["SVM", "KNN", "Decision Tree"],
                "accuracy": [svm_results['overall']['accuracy'], knn_results['overall']['accuracy'], dt_results['overall']['accuracy']],
                "precision": [svm_results['overall']['precision'], knn_results['overall']['precision'], dt_results['overall']['precision']],
                "recall": [svm_results['overall']['recall'], knn_results['overall']['recall'], dt_results['overall']['recall']],
                "f1_score": [svm_results['overall']['f1_score'], knn_results['overall']['f1_score'], dt_results['overall']['f1_score']]
            },
            "overall": best_results["overall"],
            "confusion_matrix": best_results["confusion_matrix"],
            "roc_auc": best_results["roc_auc"],
            "training_history": {
                "epochs": [1, 2, 3],
                "accuracy": [svm_results['overall']['accuracy'], knn_results['overall']['accuracy'], dt_results['overall']['accuracy']],
                "loss": [1.0 - svm_results['overall']['accuracy'], 1.0 - knn_results['overall']['accuracy'], 1.0 - dt_results['overall']['accuracy']],
                "val_accuracy": [svm_results['overall']['accuracy'], knn_results['overall']['accuracy'], dt_results['overall']['accuracy']],
                "val_loss": [1.0 - svm_results['overall']['accuracy'], 1.0 - knn_results['overall']['accuracy'], 1.0 - dt_results['overall']['accuracy']]
            }
        }
        
        with open(metrics_filepath, 'w') as f:
            json.dump(metrics_summary, f, indent=2)
            
        logs_list.append(f"[SUCCESS] All model training and evaluations compiled.")
        logs_list.append(f"[SUCCESS] Best Model: {best_name.upper()} (F1: {best_f1*100:.2f}%)")
        
        save_status(status_filepath, status="completed", progress=100, current_epoch=3, total_epochs=3,
                    metrics=best_results["overall"], history=history, logs=logs_list)
        
        print("ML Model training and evaluation successfully completed!")
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error during training: {error_msg}")
        try:
            status_err = {
                "status": "failed",
                "progress": 0,
                "error": error_msg,
                "logs": logs_list + [f"[ERROR] Exception occurred: {error_msg}"]
            }
            with open(status_filepath, 'w') as f:
                json.dump(status_err, f, indent=2)
        except Exception:
            pass
        raise e

if __name__ == '__main__':
    classes = ['Black Soil', 'Cinder Soil', 'Laterite Soil', 'Loam Soil', 'Peat Soil', 'Yellow Soil']
    try:
        train_and_evaluate(classes)
    except Exception as e:
        print(f"Run failed: {e}")
