from numpy import mat
def list2mat(list):
    m1 = [list[0], list[1], list[2], list[3]]
    m2 = [list[4], list[5], list[6], list[7]]
    m3 = [list[8], list[9], list[10], list[11]]
    m4 = [list[12], list[13], list[14], list[15]]
    matrix = mat([m1, m2, m3, m4])
    return matrix


def mat2list(matrix):
    lis = [matrix[0, 0], matrix[0, 1], matrix[0, 2], matrix[0, 3], \
           matrix[1, 0], matrix[1, 1], matrix[1, 2], matrix[1, 3], \
           matrix[2, 0], matrix[2, 1], matrix[2, 2], matrix[2, 3], \
           matrix[3, 0], matrix[3, 1], matrix[3, 2], matrix[3, 3]]
    return lis