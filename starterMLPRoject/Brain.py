#Brain class to handle all ML related tasks
# 1 > Normalize points
# 2 > Transmute normal coords into static length matrix


class Brain:

    #Method to convert input points to *scaled* coordinates AKA normalized data ready for ML
    def normalizePoints(points):
        if not points:                              #Handle empty list case
            return []
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        #Step 1 : Get min and max x and y values from the points
        xs = []
        ys = []
        for point in points:
            xs.append(point[0])                    #get x's
            ys.append(point[1])                    #get y's

        minX,maxX = min(xs),max(xs)                #get min and max x
        minY,maxY = min(ys),max(ys)                #get min and max y
        rangeX = maxX - minX                       #get range of x's
        rangeY = maxY - minY                       #get range of y's
        #(Handle 0 division case)
        if rangeY == 0:
            rangeY = 1
        if rangeX == 0:
            rangeX = 1
        #- - - - - - - - - - - - -

        for point in points:
            #Step 2: shift the points to line up with origin of the canvas (0,0)
            point[0] -= minX                      #shift x's
            point[1] -= minY                      #shift y's
            # - - - - - - - - - -
            #Step 3: Scale all the points to fit in a UNIT SQUARE [(0,0) to (1,1)]
            point[0] /= rangeX                    #scale x's
            point[1] /= rangeY                    #scale y's
        
        return points                            
    
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 

    #Method to convert normalized points to a fixed length matrix
    def genNormalMatrix(normalizedPoints, res=10):
        # Step 1 : Create a matrix of zeros
        matrix = [[0 for _ in range(res)] for _ in range(res)]
        # Step 2 : Fill the matrix with normalized points
        for point in normalizedPoints:
            x = int(point[0] * res)
            y = int(point[1] * res)
            # Ensure x and y are within bounds
            if x >= res:
                x = res - 1
            if y >= res:
                y = res - 1
             # Set the point in the matrix to 1
            matrix[y][x] = 1
        
        return matrix
    # = = = = = = = = = = = = = = = =
    #Method to visualize the matrix in console
    def visualizeMatrix(matrix):
        visualization = ""
        for row in matrix:
            visualization += " ".join(str(cell) for cell in row) + "\n"

        return visualization

    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 

    #Method to flatten the matrix into a 1D List. (Suitable for ML input)
    def flattenMatrix(matrix):
        flatVector = []
        for row in matrix:
            for cell in row:
                flatVector.append(cell)
                
        return flatVector