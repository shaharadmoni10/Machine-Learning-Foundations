import numpy as np #אני מייבא את הספריה על מנת להשתמש בפעולות אריתמטיות
x = np.random.rand(100,4) #אני מגדיר מטריצה אקראית בעלת 100 שורות ו 4 עמודות
beta = np.array([1,2,3,4]) #אני מגדיר משתנה בטא שהוא וקטור עם 4 ערכים
epsilon = np.random.randn(100, 1) #אני מגדיר משתנה אפסילון שהוא תוספת רעש אקראית 
y = (x @ beta).reshape(100, 1) + epsilon #הגדרת כפל מטריצות וסידורם כוקטור עמודה והוספת רעש 
beta_hat = ((np.linalg.inv(x.T @ x) @ x.T) @ y).flatten() #אני מחשב את האומדן שלי על פי הנוסחה כאשר אני נעזר בספריה לבצע כפל מטריצות והיפוך מטריצות
print("The true beta values are:", beta) #אני מדפיס את הערכים האמיתיים שהגדרתי לפני
print("The beta estimates are:", beta_hat) #אני מדפיס את הערכים של בטא שיצאו לפי החישוב
print ("the error is:", np.abs(beta - beta_hat)) #אני מדפיס את השגיאה בין הערכים האמיתיים למה שחישבתי

epsilon_big = np.random.randn(100, 1) * 10 #אני מגדיר משתנה אפסילון חדש עם רעש אקראי מוגבר פי 10
y_new = (x @ beta).reshape(100, 1) + epsilon_big 
beta_hat_2 = ((np.linalg.inv(x.T @ x) @ x.T) @ y_new).flatten()
print("The beta estimates with larger noise are:", beta_hat_2) 
print ("the error with larger noise is:", np.abs(beta - beta_hat_2))
print ("As we can see, increasing the noise leads to larger errors by:,", np.abs(beta - beta_hat_2) / np.abs(beta - beta_hat)) #אני מראה את הקפיצה בשגיאה