import numpy as np
from sklearn.linear_model import LinearRegression

#x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
#y = [4, 5, 20, 14, 32, 22, 38, 43]
x = [[300, 25, 250, 21], [250, 18, 325, 28]]
y = [23, 25]

x, y = np.array(x), np.array(y)
model = LinearRegression().fit(x, y)

r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)

x_new = [[3301/9, 270/9, 225, 18], [3517/9, 252/9, 300, 25]]
y_pred = model.predict(x_new)
print('predicted response:', y_pred, sep='\n')


        #data_frame_stats = pd.DataFrame(data=one_year_stats) # don't think i need this if i use sklearn

        #point_nums = data_frame_stats['points'].astype('int')
        #print(point_nums.max())
        #print(point_nums.min())