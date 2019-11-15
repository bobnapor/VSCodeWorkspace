class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

#looking for 4 points such that:
    #(x1, y1) , (x1, y2) , (x2, y1) , (x2, y2)

def count_rectangles(points):
    rectangles = []
    for point1 in points:
        x1 = point1.get_x()
        y1 = point1.get_y()
        rectangles2 = rectangles.remove(point1) #do something recursively to check this point against remaining points until i have 4 pts

