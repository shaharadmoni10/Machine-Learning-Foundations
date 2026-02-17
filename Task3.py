import numpy as np #ספריה שתעזור עם חישובים אלגברה לינארית 
import matplotlib.pyplot as plt #ספריה שתעזור עם הצגה ויזואלית של גרפים של מטלאב
import scipy.io # ספריה שתעזור לטעון קבצי מטלאב וחילוץ הקובץ
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(script_dir, 'mnist_all.mat')
mt = scipy.io.loadmat(file_path)
data_list = [] #רשימה לשמירת המידע של כל ספרה 
label_list = []#רשימה לשמירת התוויות של כל ספרה
k = 10
cost_history = [] # אני רוצה לשמור את העלות של כל האיטרציות עד רגע מסוים
epsilon = 1e-5 #סף עצירה למקרה שהעלות לא משתנה הרבה בין איטרציות

def calc_accuracy(new_labels, true_labels):
    total_correct = 0
    for i in range(10):
        indices = np.where(new_labels == i)[0]
        true_labels_in_cluster = true_labels[indices]
        if len(true_labels_in_cluster) > 0:
            total_correct += np.max(np.bincount(true_labels_in_cluster))
    return total_correct / len(true_labels)


for i in range(10): #אני רוצה לחלץ את המידע מהקובץ ואני יודע שיש 10 מפחתות ל10 מספרים
  key_data = 'train' + str(i) #מגדיר את המפתח של כל מספר
  data = mt[key_data] #חילוץ המידע של כל מספר
  data_list.append(data) #חילוץ המידע של כל מספר
  labels = np.full(data.shape[0], i) #יצירת וקטור תוויות מלא בערך של הספרה המתאימה
  label_list.append(labels) #שמירת וקטור התוויות ברשימה
  
x = np.concatenate(data_list, axis=0) #מיזוג כל המטריצות של הדאטה למטריצה אחת גדולה
y = np.concatenate(label_list, axis=0) #מיזוג כל וקטורי התוויות לוקטור אחד גדול
X = x.astype(np.float64) / 255.0 #נרצה לנרמל על מנת שנוכל לחשב את הסיגמואיד 
print("The shape of the data matrix is:", X.shape) #אני מדפיס את המטריצה לוודא שהגיוני  
print("The shape of the labels vector is:", y.shape) #אני מדפיס את וקטור התוויות לוודא שהגיוני

n_samples = X.shape[0] #מספר הדוגמאות
rand_indices = np.random.choice(n_samples, k, replace=False) #בחירת k אינדקסים אקראיים ללא חזרה
sample_images = X[rand_indices] #בחירת התמונות המתאימות לאינדקסים 
print(f"the centers of the clusters are: {sample_images.shape}")

x_square_sum = np.sum(X**2, axis=1, keepdims=True) #חישוב סכום הריבועים של כל שורה במטריצה X

for i in range (100):
  center_square_sum = np.sum(sample_images**2, axis=1) #חישוב סכום הריבועים של כל שורה במטריצה של מרכזי האשכולות
  cross_term = X @ sample_images.T #חישוב המכפלה המטריציונית בין X למרכז האשכול הנוכחי 
  sum_all = x_square_sum + center_square_sum - 2 * cross_term #חישוב המרחקים הריבועיים בין כל נקודה למרכז האשכול הנוכחי
  new_labels = np.argmin(sum_all, axis=1) #מציאת האינדקס של מרכז האשכול הקרוב ביותר לכל נקודה
  min_sum_all = np.min(sum_all, axis=1) #מציאת המרחק המינימלי לכל נקודה למרכז האשכול הקרוב ביותר
  cost = np.sum(min_sum_all) #חישוב פונקציית העלות כסכום המרחקים המינימליים
  cost_history.append(cost) #שמירת פונקציית העלות ברשימה
  new_centers = np.zeros(sample_images.shape) #יצירת מטריצה חדשה למרכזי האשכולות
  for j in range(k):     
     points_in_cluster = X[new_labels == j] #בחירת כל הנקודות ששויכו לאשכול הנוכחי
     if len(points_in_cluster) > 0: #אם יש נקודות באשכול
      new_centers[j] = np.mean(points_in_cluster, axis=0) #עדכון מרכז האשכול כממוצע הנקודות באשכול
     else:
      new_centers[j] = sample_images[j] #אם אין נקודות באשכול, נשאיר את המרכז כפי שהוא
  
  diff = np.linalg.norm(new_centers - sample_images)
  
  if diff < epsilon:
    print(f"Algorithm converged at iteration {i} with change {diff}")
    sample_images = new_centers # עדכון אחרון חשוב
    break 
  sample_images = new_centers #עדכון מרכזי האשכולות למרכזים החדשים שחושבו
  print(f"Iteration {i}: cost = {cost:.2f}")   

print(f"Accuracy: {calc_accuracy(new_labels, y):.2%}")


