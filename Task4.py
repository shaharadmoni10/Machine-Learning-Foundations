import numpy as np #יבוא לחישבים
import matplotlib.pyplot as plt #יבוא לציור
from scipy.stats import multivariate_normal #יבוא הכלי לחישוב גאוסיאן


def generate_gmm_data(n_samples=1000): #הגדרת הנקודות באופן ראשוני
    alphas = [0.3, 0.4, 0.3] #הסתברויות של כל נקודה  להשתייך למחלקה 1,2,3
    mu1 = [0, 5]; sigma1 = [[1, 0], [0, 1]] #הממוצעים והקוואריאנס של כל מחלקה
    mu2 = [5, 0]; sigma2 = [[1.5, 0], [0, 1.5]]     
    mu3 = [10, 5]; sigma3 = [[1, 0], [0, 1]]
    
    params = [    #אני יוצר רשימה של כל המחלקות על פי מבנה נתונים של מילון שיהיה לי קל לנהל את זה ולגשת לזה
        {'alpha': alphas[0], 'mu': mu1, 'sigma': sigma1},
        {'alpha': alphas[1], 'mu': mu2, 'sigma': sigma2},
        {'alpha': alphas[2], 'mu': mu3, 'sigma': sigma3}
    ]
    #יצירת רשימה של כל הנקודות והכתיבה של התשובות שנוכל להשוות לאחר מכן 
    data_points = []
    true_labels = []
    
    for i in range(n_samples): #לכל נקודה
        z_i = np.random.choice(3, p=alphas) #ההגרלה של נקודות כאשר אני מגריל את השייכות שלה ל 3 המחלקות על פי המשקלים שהבאתי לה קודם
        current_params = params[z_i] #הפרמטרים של המחלקה
        point = np.random.multivariate_normal(current_params['mu'], current_params['sigma']) #הגרלת נקודה
        data_points.append(point) #הוספת הנקודה לרשימה
        true_labels.append(z_i) #הוספת התשובה האמיתית לרשימה
        
    return np.array(data_points), np.array(true_labels) #החזרת הנקודות וההתשובות האמיתיות


class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None

    def fit(self, X):
        # אתחול
        indices = np.random.choice(X.shape[0], self.k, replace=False) #הגרלת מרכזים
        self.centroids = X[indices] #הוספת המרכזים לרשימה
        
        labels = np.zeros(X.shape[0]) #הוספת התשובה האמיתית לרשימה
        
        for _ in range(self.max_iters): #לכל נקודה
            dists = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2) #הגרלת מרכזים
            new_labels = np.argmin(dists, axis=1) #הוספת המרכזים לרשימה 
            
            if np.all(labels == new_labels):
                break
            labels = new_labels
            
            # עדכון מרכזים
            for i in range(self.k):
                points = X[labels == i]
                if len(points) > 0:
                    self.centroids[i] = np.mean(points, axis=0)
        return labels

# מחלקת GMM חדשה
class GMM:
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
    
    def fit(self, X):
        n_samples, n_features = X.shape
        
        #  אתחול פרמטרים 
        # חלוקת משקולוץ
        self.phi = np.ones(self.k) / self.k
        
        # בחירת נקודות אקראיות, ממוצעים
        indices = np.random.choice(n_samples, self.k, replace=False)
        self.mu = X[indices]
        
        # מטריצת קוואריאנס
        self.sigma = np.array([np.eye(n_features) for _ in range(self.k)])
        
        self.w = np.zeros((n_samples, self.k))
        
        for iteration in range(self.max_iters):
            # חישוב הסתברויות
            # חישוב מונה
            for j in range(self.k):
                pdf = multivariate_normal.pdf(X, mean=self.mu[j], cov=self.sigma[j])
                self.w[:, j] = pdf * self.phi[j]
            
            # נרמול השורות
            w_sum = self.w.sum(axis=1)[:, np.newaxis]
            self.w = self.w / w_sum
            
            #עדכון פרמטרים וסכום המשקולות
            N_j = self.w.sum(axis=0)
            
            #  עדכון phi 
            self.phi = N_j / n_samples
            
            new_mu = np.zeros_like(self.mu)
            for j in range(self.k):
                #  עדכון mu 
                # מכפלה סקלרית בין המשקל של הנקודה לערך שלה
                new_mu[j] = np.sum(self.w[:, j][:, np.newaxis] * X, axis=0) / N_j[j]
                
                # עדכון sigma 
                diff = X - new_mu[j] # (x - mu)
                # חישוב מטריצת השונות המשותפת המשוקללת
                weighted_diff = diff.T @ (diff * self.w[:, j][:, np.newaxis])
                self.sigma[j] = weighted_diff / N_j[j]
            
            # בדיקת התכנסות על ה-Mu
            if np.all(np.abs(new_mu - self.mu) < self.tol):
                self.mu = new_mu
                break
            self.mu = new_mu

        # החזרת התגית בעלת ההסתברות הגבוהה ביותר לכל נקודה
        return np.argmax(self.w, axis=1)

#  פונקציית עזר לחישוב הצלחה 
def calculate_accuracy(pred_labels, true_labels):
    """
    ממפה את הקלאסטרים לתגיות האמיתיות ובודק דיוק
    """
    mapping = {}
    # עוברים על כל קלאסטר שנמצא
    for i in np.unique(pred_labels):
        # מוצאים את התגיות האמיתיות של הנקודות ששויכו לקלאסטר הזה
        true_labels_in_cluster = true_labels[pred_labels == i]
        if len(true_labels_in_cluster) > 0:
            # הרוב קובע: התגית הנפוצה ביותר היא התגית של הקלאסטר
            most_common = np.bincount(true_labels_in_cluster).argmax()
            mapping[i] = most_common
        else:
            mapping[i] = -1
            
    # ממירים את התחזיות לתגיות האמיתיות לפי המיפוי
    mapped_preds = np.array([mapping[l] for l in pred_labels])
    return np.mean(mapped_preds == true_labels)


# יצירת נתונים
X, y_true = generate_gmm_data()

# הרצת K-Means 
kmeans = KMeans(k=3)
kmeans_labels = kmeans.fit(X)
kmeans_acc = calculate_accuracy(kmeans_labels, y_true)

# הרצת GMM 
gmm = GMM(k=3)
gmm_labels = gmm.fit(X)
gmm_acc = calculate_accuracy(gmm_labels, y_true)

#  הדפסת תוצאות והשוואה 
print(f"K-Means Accuracy: {kmeans_acc:.2%}")
print(f"GMM Accuracy:     {gmm_acc:.2%}")

#  ויזואליזציה להשוואה
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_true, s=10)
plt.title('True Data')

plt.subplot(1, 3, 2)
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, s=10, cmap='viridis')
plt.title(f'K-Means (Acc: {kmeans_acc:.2%})')

plt.subplot(1, 3, 3)
plt.scatter(X[:, 0], X[:, 1], c=gmm_labels, s=10, cmap='plasma')
plt.title(f'GMM (Acc: {gmm_acc:.2%})')

plt.show()