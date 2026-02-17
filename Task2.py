import numpy as np #ספריה שתעזור עם חישובים אלגברה לינארית 
import matplotlib.pyplot as plt #ספריה שתעזור עם הצגה ויזואלית של גרפים של מטלאב
import scipy.io # ספריה שתעזור לטעון קבצי מטלאב וחילוץ הקובץ
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(script_dir, 'mnist_all.mat')
mt = scipy.io.loadmat(file_path)

def sigmoid (z): 
  return 1 / (1 + np.exp(-z)) #הגדרת פונקציית הסיגמואיד    
def loss_cost (x, y, w): #הגדרת פונקציית העלות
  N = len(y) #מספר הדוגמאות 
  z = x @ w #כפל מטריצות בין המטריצה של הדאטה למטריצה של המשקלים   
  h = sigmoid(z)    #הפעלת פונקציית הסיגמואיד על התוצאה של כפל המטריצות 
  sum_loss = y @ np.log(h + 1e-15) + (1 - y) @ np.log(1 - h + 1e-15)    #חישוב פונקציית הלוס לפי הנוסחה
  return (1/N) * sum_loss #החזרת הערך הממוצע של פונקציית הלוס
  
def gradient_ascent(x, y, w, learning_rate, num_iterations):
    N = len(y) #מספר הדוגמאות 
    cost_history = [] #רשימה לשמירת היסטוריית פונקציית העלות   
    for i in range(num_iterations): #לולאה על מספר האיטרציות
        z = x @ w #כפל מטריצות בין המטריצה של הדאטה למטריצה של המשקלים   
        h = sigmoid(z)    #הפעלת פונקציית הסיגמואיד על התוצאה של כפל המטריצות 
        gradient = (1/N) * (x.T @ (y- h)) #חישוב הגרדיאנט לפי הנוסחה
        w = w + learning_rate * gradient #עדכון המשקלים לפי הגרדיאנט ולפי קצב הלמידה
        cost = loss_cost(x, y, w) #חישוב פונקציית העלות לאחר עדכון המשקלים
        cost_history.append(cost) #שמירת פונקציית העלות ברשימה 
        if i % 100 == 0: #כל 100 איטרציות נדפיס את פונקציית העלות כדי לראות שהולך לכיוון הנכון
         print(f"iteration {i}: cost = {cost}")
    return w, cost_history #החזרת המשקלים המעודכנים והיסטוריית פונקציית העלות




for i in range(10): #אני רוצה לחלץ את המידע מהקובץ ואני יודע שיש 10 מפחתות ל10 מספרים
  key_data = 'train' + str(i) #מגדיר את המפתח של כל מספר
  data = mt[key_data] #חילוץ המידע של כל מספר
  image_vector = data [0] #אני מכניס למשתנה תמונה שהוא וקטור שורה את המידע שאספתי. כאשר יש אלפי שורות שבונות את הספרות בצורה קצת אחרת כל פעם
  image_matrix = image_vector.reshape(28,28) #אני הופך את וקטור השורה כמו שביקשו מאיתנו למטריצה 28 על 28 
  plt.imshow(image_matrix, cmap='gray') #אנחנו מציגים בשחור לבן
  plt.show() #הצגת הגרף 

# כעת נאמן את המודל שלנו
data_digits1 = mt['train1'] #אני בוחר את הספרה 1 כדי לאמן את המודל שלי
data_digits2 = mt['train2'] #אני בוחר את הספרה 2 כדי לאמן את המודל שלי
labels_digits1=np.zeros(len(data_digits1)) #אני מגדיר את התוויות של הספרה 1 להיות 0
labels_digits2=np.ones(len(data_digits2)) #אני מגדיר את התוויות של הספרה 2 להיות 1
x = np.concatenate((data_digits1, data_digits2))
y = np.concatenate((labels_digits1, labels_digits2))
print("The shape of the data matrix is:", x.shape) #אני מדפיס את המטריצה לוודא שהגיוני
print("The shape of the labels vector is:", y.shape) 

x = x / 255.0 #נרצה לנרמל על מנת שנוכל לחשב את הסיגמואיד
ones_column = np.ones((x.shape[0], 1)) #אני יוצר עמודת אחדות כדי להוסיף למטריצה של הדאטה
x = np.hstack((ones_column, x)) #אני מוסיף את עמודת
learning_rate = 0.1 #קצב הלמידה
num_iterations = 1000 #מספר האיטרציות  
w = np.zeros(x.shape[1]) #אתחול וקטור המשקלים באפסים בהתאם למספר התכונות במטריצה של הדאטה  
print ("Let's start the show:")
w, cost_history = gradient_ascent(x, y, w, learning_rate, num_iterations) #קריאה לפונקציית הגרדיאנט דסנט
print("The show is over!")
print("The Final Cost is:", cost_history[-1]) #הדפסת פונקציית העלות הסופית לאחר כל האיטרציות

#caclulate accuracy at final test after we learn the weights and minimize the loss function
accuracy_z = x @ w #אני כופל מטריצות של דאטה ומשקלים כמו בנוסחה
accuracy_h =  sigmoid(accuracy_z) #אני מפעיל את הפונקציה של סיגמויד כדי להפוך את זה להסתברות בין 0 ל 1
predictions = accuracy_h >= 0.5 #אני מגדיר את זה ככה שיהיה סף לתשובה 0 או 1, אצלי זה טריוויאלי שיהיה 0.5
accuracy = np.mean(predictions == y) #אני לוקח את המטריצה של התשובות הנכונות שלי, Y, ומבצע ממוצע על החיזוי שהיה שווה ל Y 
print(f"The accuracy of the model is: {accuracy*100:.2f}%") 