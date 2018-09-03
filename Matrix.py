class MatrixIndexError(Exception):
    '''An attempt has been made to access an invalid index in this matrix'''


class MatrixDimensionError(Exception):
    '''An attempt has been made to perform an operation on this matrix which
    is not valid given its dimensions'''


class MatrixInvalidOperationError(Exception):
    '''An attempt was made to perform an operation on this matrix which is
    not valid given its type'''


class MatrixNode():
    '''A general node class for a matrix'''

    def __init__(self, contents, right=None, down=None,row=None,col=None):
        '''(MatrixNode, obj, MatrixNode, MatrixNode) -> NoneType
        Create a new node holding contents, that is linked to right
        and down in a matrix
        '''
        self._contents = contents
        self._right = right
        self._down = down
        self._col = col
        self._row = row

    def __str__(self):
        '''(MatrixNode) -> str
        Return the string representation of this node
        '''
        return str(self._contents)

    def get_contents(self):
        '''(MatrixNode) -> obj
        Return the contents of this node
        '''
        return self._contents

    def set_contents(self, new_contents):
        '''(MatrixNode, obj) -> NoneType
        Set the contents of this node to new_contents
        '''
        self._contents = new_contents

    def get_right(self):
        '''(MatrixNode) -> MatrixNode
        Return the node to the right of this one
        '''
        return self._right

    def set_right(self, new_node):
        '''(MatrixNode, MatrixNode) -> NoneType
        Set the new_node to be to the right of this one in the matrix
        '''
        self._right = new_node

    def get_down(self):
        '''(MatrixNode) -> MatrixNode
        Return the node below this one
        '''
        return self._down

    def set_down(self, new_node):
        '''(MatrixNode, MatrixNode) -> NoneType
        Set new_node to be below this one in the matrix
        '''
        self._down = new_node
        
    def set_col(self,val):
        '''(MatrixNode,int) -> NoneType
        Set the nodes col number
        '''
        self._col = val
    
    def get_col(self):
        '''(MatrixNode) -> int
        Get the nodes col number
        '''
        return self._col
    
    def set_row(self,val):
        '''(MatrixNode,int) -> NoneType
        Set the nodes row number
        '''
        self._row = val
        
    def get_row(self):
        '''(MatrixNode) -> int
        Get the nodes row number
        '''
        return self._row    


class Matrix():
    '''A class to represent a mathematical matrix'''

    def __init__(self, m, n, default=0):
        '''(Matrix, int, int, float) -> NoneType
        Create a new m x n matrix with all values set to default
        '''
        # make the head
        self._head = MatrixNode(None)
        # set the number of rows
        self._rows = m
        # set the number of cols
        self._cols = n
        # set the default
        self._default = default        
    
    def get_intersection_node(self,i,j):
        '''(Matrix,int,int) -> MatrixNode
        Returns the node at the given location
        '''
        # check for index errors
        self.check_for_index_errors(i,j)
        # make curr the head
        curr = self._head
        # get to the right row
        # check if curr is right row index
        while (curr != None and 
               curr.get_contents() != i):
            # go down one
            curr = curr.get_down()
        # look for our value with while loop
        # check if we have the right column or
        # if we're at the end
        while (curr != None  and curr.get_col() != j):
            # go to the next node
            curr = curr.get_right()
        # return the node
        return curr

    def get_val(self, i, j):
        '''(Matrix, int, int) -> float
        Return the value of m[i,j] for this matrix m
        '''
        # get the intersection node
        curr = self.get_intersection_node(i,j)
        # check if its there
        if (curr == None):
            # give back the default value
            result = self._default
        # if we actuall find a node
        else:
            # get the value at that node
            result = curr.get_contents()
        # return whatever we got
        return result

    def set_val(self, i, j, new_val):
        '''(Matrix, int, int, float) -> NoneType
        Set the value of m[i,j] to new_val for this matrix m
        '''
        # check for index errors
        self.check_for_index_errors(i,j)        
        # instantiate our new node for later on
        new_node = None
        # first we'll make sure its not the default val
        if (new_val != self._default):
            # start at the head
            row = self._head
            col = self._head
            # first were gunna look at the row index
            # check if the row index node exists
            # go down from the head, see if we find row index node
            # stop when we reach none, we find the node, or 
            # if we find a row thats larger than what were looking for
            while ((row.get_down() is not None) 
                   and row.get_down().get_contents() < i):
                # go down to the next row node
                row = row.get_down()
            # if we dont need to make a new node
            if (row.get_down() is not None and 
                row.get_down().get_contents() == i):
                # set it to the the row node
                new_row = row.get_down()
            # if we need to make a new node
            else:
                # make a new row node
                new_row = MatrixNode(i)
                # set the node below it, to the node that is
                # below the current
                new_row.set_down(row.get_down())
                # put new row below the old row
                row.set_down(new_row)
            # now were gunna do the same thing, but with the col
            # check if the row index node exists
            # go right from the head, see if we find colindex node
            # stop when we reach none, we find the node, or 
            # if we find a col thats larger than what were looking for
            while ((col.get_right() is not None) 
                   and col.get_right().get_contents() < j):
                # go right to the next col node
                col = col.get_right()
            # if we dont need to make a new node
            if (col.get_right() is not None and 
                col.get_right().get_contents() == i):
                # set it to the the col node
                new_col = col.get_right()
            # if we need to make a new node
            else:
                # make a new row node
                new_col = MatrixNode(j)
                # set the node below it, to the node that is
                # below the current
                new_col.set_right(col.get_right())
                # put new row below the old row
                col.set_right(new_col)
            # now everythings in order
            # make our new node
            # start at the row
            row_curr = new_row
            # go through it in same fashion
            while (row_curr.get_right() is not None and
                   row_curr.get_right().get_col() < j ):
                # go to the next one
                row_curr = row_curr.get_right()
            # if the node is already there
            if (row_curr.get_right() is not None and 
                row_curr.get_right().get_col() == j):
                # go to that node
                row_curr = row_curr.get_right()
                # set the contents
                row_curr.set_contents(new_val)
            # if its not already there
            else:
                # make a new node
                new_node = MatrixNode(new_val,None,None,i,j)
                # set new node to old nodes right
                new_node.set_right(row_curr.get_right())
                # make new node right of curr
                row_curr.set_right(new_node)
                # set the value at that node
                new_node.set_contents(new_val)
            # we only care about this if we made a new node
            if (new_node != None):
                # link it to the column in same fashion
                # start at new col
                col_curr = new_col
                # check if either we reach the end of the col
                # or were one before the index
                while (col_curr.get_down() is not None and
                       col_curr.get_down().get_row() != i-1):
                    # go to the next one
                    col_curr = col_curr.get_down()
                # if we reach the end
                if (col_curr.get_down() == None):
                    # make new_node point to the none
                    new_node.set_down(col_curr.get_down())
                    # put new node under curr
                    col_curr.set_down(new_node)
                # if we're above the row i-1
                else:
                    # go to i-1th row
                    col_curr = col_curr.get_down()
                    # make new_node point to curr's down node
                    new_node.set_down(col_curr.get_down())
                    # put new node under curr
                    col_curr.set_down(new_node)
        # if its zero
        else:
            # get the node there
            my_node = self.get_intersection_node(i,j)
            # check if that node already exists
            if (my_node is not None):
                # delete the node
                # first get the node to the left of it
                # instantiate column value
                col = j-1
                # try and find the node that exists to the left
                while (col>=0 and self.get_intersection_node(i,col) is None):
                    # go left one more
                    col-=1
                # if the next left node is the index node
                if (col<0):
                    # start at the head
                    curr = self._head
                    # go down to node
                    while (curr.get_contents()!=i):
                        # go down one
                        curr = curr.get_down()
                    # set that to left node
                    left = curr
                    # if the left node is in the matrix
                else:
                    # set the left node
                    left = self.get_intersection_node(i,col)
                # same for the one above it
                # instantiate row val
                row = i-1
                while (row>=0 and self.get_intersection_node(row,j) is None):
                    # go up more
                    row-=1
                # check if left node is index node
                if (row<0):
                    # start at the head
                    curr = self._head
                    # go to the right to index node
                    while (curr.get_contents() != j):
                        # go to the right
                        curr = curr.get_right()
                    # set that to up node
                    up = curr
                #if the node is in the matrix
                else:
                    # set the node value
                    up = self.get_intersection_node(row,j)
                # now delete the node
                # lefts right is my_node's right
                left.set_right(my_node.get_right())
                # ups down is my_nodes down
                up.set_down(my_node.get_down())
        # that was a lot of work
        # very tired
        
    def get_row(self, row_num):
        '''(Matrix, int) -> OneDimensionalMatrix
        Return the row_num'th row of this matrix
        '''
        # check for index errors
        self.check_for_index_errors(row_num,self._cols)
        # make a oneD to return
        oneD = OneDimensionalMatrix(self._cols,True)
        # clone the row
        # go through row with for loop
        for i in range(self._cols):
            # check if there actually is a value
            if (self.get_val(row_num,i) != 0):
                # set the value in the oneD
                oneD.set_item(i,self.get_val(row_num,i))
        # send back the oneD
        return oneD
        

    def set_row(self, row_num, new_row):
        '''(Matrix, int, OneDimensionalMatrix) -> NoneType
        Set the value of the row_num'th row of this matrix to those of new_row
        '''
        # check for index erros
        self.check_for_index_errors(row_num,self._cols)        
        # since its 1xn, go through cols
        for i in range(new_row.get_cols()):
            # set the value to set row, in the matrix
            self.set_val(row_num,i,new_row.get_item(i))

    def get_col(self, col_num):
        '''(Matrix, int) -> OneDimensionalMatrix
        Return the col_num'th column of this matrix
        '''
        # check for index errors
        self.check_for_index_errors(self_rows,col_num)        
        # make a oneD to return
        oneD = OneDimensionalMatrix(self._rows,True)
        # clone the row
        # go through row with for loop
        for i in range(self._rows):
            # check if there actually is a value
            if (self.get_val(i,col_num) != 0):
                # set the value in the oneD
                oneD.set_item(i,self.get_val(i,col_num))
        # send back the oneD
        return oneD     

    def set_col(self, col_num, new_col):
        '''(Matrix, int, OneDimensionalMatrix) -> NoneType
        Set the value of the col_num'th column of this matrix to those of new_row
        '''
        # check for index errors
        self.check_for_index_errors(self_rows,col_num)        
        # since its 1xn, go through cols
        for i in range(new_col.get_rows()):
            # set the value to set col, in the matrix
            self.set_val(i,col_num,new_col.get_item(i))              

    def swap_rows(self, i, j):
        '''(Matrix, int, int) -> NoneType
        Swap the values of rows i and j in this matrix
        '''
        # check for index errors
        self.check_for_index_errors(i,j)       
        # get the two rows in oneD
        first = self.get_row(i)
        second = self.get_row(j)
        # now set the rows to each other
        self.set_row(j,first)
        self.set_row(i,second)

    def swap_cols(self, i, j):
        '''(Matrix, int, int) -> NoneType
        Swap the values of columns i and j in this matrix
        '''
        # check for index errors
        self.check_for_index_errors(i,j)        
        # get the two rows in oneD
        first = self.get_col(i)
        second = self.get_col(j)
        # set the cols to each other
        self.set_col(j,first)
        self.set_col(i,second)

    def add_scalar(self, add_value):
        '''(Matrix, float) -> NoneType
        Increase all values in this matrix by add_value
        '''
        # go through all the rows first
        for i in range(self._rows):
            # go through all the cols
            for j in range(self._cols):
                # check for type error
                if (type(self.get_val(i,j)) != 
                    type(add_value) and 
                    self.get_val(i,j) is not self._default):
                    # raise the error
                    raise MatrixInvalidOperationError
                # add the scalar to each
                self.set_val(i,j,self.get_val(i,j)+add_value)
                

    def subtract_scalar(self, sub_value):
        '''(Matrix, float) -> NoneType
        Decrease all values in this matrix by sub_value
        '''
        # check for operation errors
        self.check_for_operation_errors(sub_value)        
        # go through all the rows first
        for i in range(self._rows):
            # go through all the cols
            for j in range(self._cols):
                # add the scalar to each
                self.set_val(i,j,self.get_val(i,j)-sub_value)        

    def multiply_scalar(self, mult_value):
        '''(Matrix, float) -> NoneType
        Multiply all values in this matrix by mult_value
        '''       
        # go through all the rows first
        for i in range(self._rows):
            # go through all the cols
            for j in range(self._cols):
                # check for type error
                if (type(self.get_val(i,j)) != 
                    type(add_value) and 
                    self.get_val(i,j) is not None):
                    # raise the error
                    raise MatrixInvalidOperationError                
                # add the scalar to each
                self.set_val(i,j,self.get_val(i,j)*mult_value)        

    def add_matrix(self, adder_matrix):
        '''(Matrix, Matrix) -> Matrix
        Return a new matrix that is the sum of this matrix and adder_matrix
        '''
        # check for dimension errors
        if (self._rows != adder_matrix.get_rows() or
            self._cols != adder_matrix.get_cols()):
            # raise the error
            raise MatrixDimensionError
        # make a matrix to return
        ret = Matrix(self._rows,self._cols)
        # go through rows
        for i in range(self._rows):
            # go through cols
            for j in range(self._cols):
                # add them to the ret
                ret.set_val(i,j,self.get_val(i,j) +
                            adder_matrix.get_val(i,j))
        # return the matrix
        return ret
    
    def print_matrix(self):
        '''(Matrix) -> str
        Prints out the matrix graphically,
        for testing purposes
        '''
        # instantiate a string
        mat = ''
        # go throught the rows
        for j in range(self._rows):
            # go through the cols
            for i in range(self._cols):
                # add the next value in the row to string
                mat+='  ' + str(self.get_val(j,i))
                # if its the last value in the row
                if (i==self._cols-1):
                    # print out the row
                    print(mat)
            # set it blank for the next row
            mat = ''
            
    def multiply_matrix(self, mult_matrix):
        '''(Matrix, Matrix) -> Matrix
        Return a new matrix that is the product of this matrix and mult_matrix
        '''
        # make sure matrices are same dimensions
        if (self._cols != mult_matrix.get_rows()):
            # raise an error
            raise MatrixDimensionError
        # instantiate matrix to multiply
        ret = Matrix(self._rows,mult_matrix.get_cols())
        # go through rows in self
        for i in range(self._rows):
            # go through cols of mult_matrix
            for j in range(mult_matrix.get_cols()):
                # now rows of mult_matrix
                for k in range(mult_matrix.get_rows()):
                    # set the value
                    ret.set_val(i,j,ret.get_val(i,j) + (self.get_val(i,k) *
                                     mult_matrix.get_val(k,j)))
        # return the next multiplied matrix
        return ret

    def get_head(self):
        '''(Matrix) -> MatrixNode
        Returns the head of the matrix
        '''
        return self._head
    
    def get_rows(self):
        '''(Matrix) -> int
        Returns number of rows in the matrix
        '''
        return self._rows
    
    def get_cols(self):
        '''(Matrix -> int
        Returns number of cols in the matrix
        '''
        return self._cols
    
    def check_for_index_errors(self,rows,cols):
        '''(int,int) -> NoneType
        Just a method to check for index errors
        '''
        # check for index errors
        if (rows<0 or rows>self._rows or
            cols<0 or cols>self._cols):
            # raise the error
            raise MatrixIndexError
        
    def check_for_operation_errors(self,val):
        '''(any) -> NoneType
        Makes sure we have an int or a float
        '''
        # check for operation errors
        if type(add_value) != int and type(add_value) != float:
            # raises the error
            raise MatrixInvalidOperationError        


class OneDimensionalMatrix(Matrix):
    '''A 1xn or nx1 matrix.
    (For the purposes of multiplication, we assume it's 1xn)'''
    
    def __init__(self,index,default=0):
        ''' (OneDimensionalMatrix,int,bool) -> NoneTyoe
        Make a new 1xn or nx1 matrix
        '''
        # set the class variable
        self._index = index
        # use the matrix init
        Matrix.__init__(self,1,index)

    def get_item(self, i):
        '''(OneDimensionalMatrix, int) -> float
        Return the i'th item in this matrix
        '''
        # use matrix get val
        ret = Matrix.get_val(self,0,i)
        # return the value
        return ret
        
    def set_item(self, i, new_val):
        '''(OneDimensionalMatrix, int, float) -> NoneType
        Set the i'th item in this matrix to new_val
        '''
        # use matrix set val
        Matrix.set_val(self,0,i,new_val)


class SquareMatrix(Matrix):
    '''A matrix where the number of rows and columns are equal'''
    
    def __init__(self,index,default=0):
        '''(SquareMatrix,int) -> NoneType
        Makes a new square matrix given one
        dimension as cols and rows are the same
        '''
        Matrix.__init__(self,index,index,default=0)

    def transpose(self):
        '''(SquareMatrix) -> NoneType
        Transpose this matrix
        '''
        # make an empty matrix
        ret = Matrix(self._cols,self._rows)
        # go through rows
        for i in range(self._rows):
            # go through cols
            for j in range(self._cols):
                # apply flipped value to ret
                ret.set_val(i,j,self.get_val(j,i))
        # return good ol ret
        return ret

    def get_diagonal(self):
        '''(Squarematrix) -> OneDimensionalMatrix
        Return a one dimensional matrix with the values of the diagonal
        of this matrix
        '''
        # make a oneD to return
        ret = OneDimensionalMatrix(self._rows)
        # go through rows and cols
        for i in range(self._cols):
            # get the value of the diagonal
            val = self.get_val(i,i)
            # set that to the value in the oneD
            ret.set_item(i,val)
        # return the oneD matrix
        return ret

    def set_diagonal(self, new_diagonal):
        '''(SquareMatrix, OneDimensionalMatrix) -> NoneType
        Set the values of the diagonal of this matrix to those of new_diagonal
        '''
        # go through the rows & cols (same number) 
        for i in range(self._cols):
            # get the value for the diagonal
            val = new_diagonal.get_item(i)
            # set the value on the diagonal
            self.set_val(i,i,val)


class SymmetricMatrix(SquareMatrix):
    '''A Symmetric Matrix, where m[i, j] = m[j, i] for all i and j'''
    
    def set_val(self,m,n,val):
        '''(SymmetricMatrix,int,int,float) -> NoneType
        Sets the value not only at the given index,
        but also the symmetric index as well
        '''
        # do the regular set
        Matrix.set_val(self,m,n,val)
        # if its not on the diagonal
        if (m!=n):
            # fix the symmetric index
            Matrix.set_val(self,n,m,val)


class DiagonalMatrix(SquareMatrix, OneDimensionalMatrix):
    '''A square matrix with 0 values everywhere but the diagonal'''
    
    def __init__(self,index,val):
        '''(DiagonalMatrix,int,int,float) -> NoneType
        Make a new SquareMatrix with the given value
        as every value on the diagonal
        '''
        # start with a regular Square init
        SquareMatrix.__init__(self,index,default = 0)
        # go through the diagonal (same val)
        for i in range(self._cols):
            # set the diagonal
            self.set_val(i,i,val)

class IdentityMatrix(DiagonalMatrix):
    '''A matrix with 1s on the diagonal and 0s everywhere else'''
    
    def __init__(self,index):
        '''(IdentityMatrix,int)
        Makes a new square matrix with 1's along the 
        diagonal
        '''
        # use the diagonal init with 1 as value
        DiagonalMatrix.__init__(self,index,1)