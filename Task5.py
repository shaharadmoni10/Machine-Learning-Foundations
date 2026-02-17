import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io 
import os
from sklearn.decomposition import PCA as SklearnPCA 


def load_data():
    # חישוב הנתיב לקובץ שנמצא באותה תיקייה כמו הסקריפט
    script_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(script_dir, 'facesData.mat')
    
    try:
        mt = scipy.io.loadmat(file_path)
        
        # חיפוש אוטומטי של שם המשתנה 
        variable_name = [k for k in mt.keys() if not k.startswith('__')][0]
        print(f"DEBUG: Found variable name in file: '{variable_name}'")
        
        data = mt[variable_name]
        # בדיקה והיפוך במידת הצורך
        # אנו מצפים ל-165 דוגמאות (15 אנשים * 11 תמונות)
        if data.shape[0] != 165:
            if data.shape[1] == 165:
                print(f"DEBUG: Transposing data from {data.shape} to (165, 1024)")
                data = data.T
            else:
                print(f"WARNING: Data shape {data.shape} is unexpected. Check your file.")
            
        return data
        
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        print("Please make sure the .mat file is in the same folder as this script.")
        return None
    except IndexError:
        print("Error: Could not find any data variable in the .mat file.")
        return None


#חלוקה לסט אימון ולסט מבחן
def split_data(data, n_people=15, n_train=8, n_test=3):
    train_set = []
    test_set = []
    train_labels = []
    test_labels = []

    for i in range(n_people):
        start_idx = i * (n_train + n_test)
        person_imgs = data[start_idx : start_idx + n_train + n_test]
        
        train_imgs = person_imgs[:n_train]
        test_imgs = person_imgs[n_train:]
        
        train_set.append(train_imgs)
        test_set.append(test_imgs)
        
        train_labels.extend([i] * n_train)
        test_labels.extend([i] * n_test)

#החזרת סט אימון וסט מבחן בצורה של מטריצת שורות
    return (np.vstack(train_set), np.array(train_labels), 
            np.vstack(test_set), np.array(test_labels))

# נרצה לחשב רק את 120 הוקטורים הראשונים שרלוונטיים ולא את האפסים

def compute_pca(X_train):
    # חישוב ממוצע
    mean_face = np.mean(X_train, axis=0)
    X_centered = X_train - mean_face
    
    # שימוש ב SVD לחישוב יציב יותר
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    eigenfaces = Vt
    
    return mean_face, eigenfaces, X_centered

#זיהוי תמונה חדשה והטלה למרחב החדש

def project_and_predict(X_train_centered, X_test, mean_face, eigenfaces, K, train_labels):
    # בחירת K האיברים הראשונים
    components = eigenfaces[:K]
    
    # הטלת נתוני האימון למרחב החדש
    train_projections = np.dot(X_train_centered, components.T)
    
    # הטלת נתוני הבדיקה
    X_test_centered = X_test - mean_face
    test_projections = np.dot(X_test_centered, components.T)
    
    predictions = []
    for test_vec in test_projections:
        # מציאת השכן הקרוב ביותר (מרחק אוקלידי)
        distances = np.linalg.norm(train_projections - test_vec, axis=1)
        nearest_idx = np.argmin(distances)
        predictions.append(train_labels[nearest_idx])
        
    return np.array(predictions)

#הצגת נתונים    
# הגדרות
H, W = 32, 32
N_PEOPLE = 15
IMGS_PER_PERSON = 11

print("--- Starting Face Recognition Script ---")

# טעינת נתונים
data = load_data()

if data is not None:
    print(f"Data loaded successfully. Shape: {data.shape}")
    
    # חלוקה ל-Training ו-Test
    X_train, y_train, X_test, y_test = split_data(data)

    # ביצוע PCA ידני
    print("Computing PCA...")
    mean_face, eigenfaces, X_train_centered = compute_pca(X_train)

    # מציגים 20 Eigenfaces
    print("Displaying Mean Face and Eigenfaces...")
    plt.figure(figsize=(12, 6))
    
    plt.subplot(3, 7, 1)
    # שימוש ב-order='F' קריטי לקבצי מטלאב
    plt.imshow(mean_face.reshape(H, W, order='F'), cmap='gray')
    plt.title("Mean Face")
    plt.axis('off')

    for i in range(20):
        plt.subplot(3, 7, i + 2)
        plt.imshow(eigenfaces[i].reshape(H, W, order='F'), cmap='gray')
        plt.title(f"EF {i+1}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    #הערכת ביצועים
    k_values = [1, 5, 10, 20, 30, 40, 50, 100, 120]
    accuracies = []

    print("\nCalculating accuracy for different K values:")
    for k in k_values:
        preds = project_and_predict(X_train_centered, X_test, mean_face, eigenfaces, k, y_train)
        acc = np.mean(preds == y_test)
        accuracies.append(acc)
        print(f"  K={k}: Accuracy={acc:.2f}")

    #גרף דיוק
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, accuracies, marker='o')
    plt.title("Classification Rate vs K")
    plt.xlabel("Number of Eigenfaces (K)")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()

    # השוואה לספריית Sklearn
    print("\n--- Validation with Sklearn ---")
    print(f"Manual PCA shape: {eigenfaces.shape}")
    pca_lib = SklearnPCA(n_components=120)
    pca_lib.fit(X_train)
    print(f"Sklearn PCA shape: {pca_lib.components_.shape}")
    print("\nDone.")