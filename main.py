import numpy as np
import pandas
import galois
import re

class GF2Mat(pandas.DataFrame):
    # degree = 2

    def drop_zeros_only_columns(self) -> pandas.DataFrame:
        # https://stackoverflow.com/questions/21164910/how-do-i-delete-a-column-that-contains-only-zeros-in-pandas
        return  self.loc[:, (self != 0).any(axis=0)]


    def galois_row_reduce(self) -> pandas.DataFrame:
        np_arr: np.array = pandas.DataFrame.to_numpy(self, dtype=np.uint8)
        gf2 = galois.GF2(np_arr, dtype=np.uint8)
        gf2 = gf2.row_reduce()
        return pandas.DataFrame(gf2, columns=self.columns)


    def has_solution(self) -> bool:
        for row in self.values:
            if row[-1] == 1:
                if sum(row) - row[-1] == 0:
                    return False        
        return True  


    def is_valid_solution(self, solution: list[int]) -> bool:
        solution_row = self._get_solution_row(solution)
        if (len(solution_row) == 0):
            return False # throw ex?
        return ((self & solution_row).sum(axis=1) % 2 == 0).min()


    def _get_solution_row(self, solution: list[int]) -> list[int]:
        if (len(solution) != var_count):
            return []
        col_names = self.iloc[0].axes[0]
        solution_row = []
        for col_name in col_names:
            solution_row.append(1)
            if col_name == (-1,):
                continue
            for var in col_name:
                solution_row[-1] &= solution[var-1]
        return solution_row


    # def solve(self) -> dict[str, int]:
    #     matrix = self
    #     # matrix = GF2Mat(matrix.drop_zeros_only_columns())
    #     matrix = GF2Mat(matrix.galois_row_reduce())
    #     matrix = GF2Mat(matrix._xor_same_indices())

    #     variables: dict[str, int] = matrix._set_last_half_variables_to_zeros()
    #     self = matrix
    #     matrix = GF2Mat(matrix._calculate_zero_columns(variables))
    #     return


    def swap_columns(self, index1: int, index2: int) -> pandas.DataFrame:
        self.loc[index1], self.loc[index2] = self.loc[index2], self.loc[index1]




    def _xor_same_indices(self) -> pandas.DataFrame:
        # VERIFY IF THE COLUMN EXISTS FIRST
        for x in range(2, degree+1):
            for index in range(1, var_count+1):
                index =  (index, )
                self[index] = self[index * x] ^ self[index]
                self = self.drop(labels=index * x, axis=1)
            return self
        

    # def _set_last_half_variables_to_zeros(self) -> dict[str, int]:
    #     variables = dict()
    #     keys = self[self.columns[-(var_count//2)-1:]].keys().array
    #     for key in keys:
    #         variables[key] = 0
    #     variables.pop("1")
    #     return variables


    # def _calculate_zero_columns(self, variables: dict[str, int]) -> pandas.DataFrame:
    #     matrix = self
    #     for column in matrix.columns:
    #         for variable in variables:
    #             variable = variable.replace("x", "")
    #             if column.endswith(variable) or column.find(variable+"_") > -1:
    #                 matrix["1"] = matrix["1"] ^ matrix[column] # not sure if we need to do that
    #                 matrix = matrix.drop(labels=column, axis=1)
    #                 break
    #     return matrix


    def _get_next_combination(self):
        solution = np.array([0] * var_count)
        yield solution
        while solution.min() != 1:
            shift = 0
            while True:
                if solution[(var_count-1)-shift] == 1:
                    solution[(var_count-1)-shift] = 0
                    shift += 1
                    continue
                solution[[(var_count-1)-shift]] = 1
                break
            yield solution


    def get_all_solutions(self) -> list[list[int]]:
        solutions = []
        for combination in self._get_next_combination():
            if self.is_valid_solution(combination):
                solutions.append(combination.copy())
        return solutions
    

    def generate_new_rows(self, increase_degree: bool = True) -> pandas.DataFrame:
        rows = len(self)
        for i in range(rows):
            for j in range(var_count):
                self = self.generate_new_row(j+1, i, increase_degree)
        return self


    def generate_new_row(self, variable: int, row_index: int, increase_degree: bool = False) -> pandas.DataFrame:
        global degree # TODO remove when not needed
        new_row = self.iloc[row_index].copy()
        for column, value in new_row.items(): #if columns are swapped it may not work
            if (value == 1):
                if (variable in column):
                    continue
                if (len(column) >= degree):
                    if (increase_degree):
                        degree += 1
                    else:
                        return self
                if (len(column) == 1 and column[0] == -1):
                    new_index = (variable,)
                else:
                    new_index = tuple(sorted(list((variable,) + column)))
                if not new_index in new_row.keys():
                    self.insert(0, new_index, 0)
                    temp = pandas.DataFrame(new_row).T
                    temp.insert(0, new_index, 1)
                    new_row = pandas.Series(temp.iloc[0], dtype=pandas.Int8Dtype())
                else:
                    new_row[new_index] = new_row[new_index] ^ 1
                new_row[column] = 0
        self.loc[len(self)] = new_row
        return self
    

    def drop_empty_rows(self) -> pandas.DataFrame:
        return self.loc[~(self == 0).all(axis=1)]


def generate_col_names(n: int) -> list[str]:
    res = []
    for i in range(1, n+1):
        for j in range(1, i+1):
            res.append((j, i))
    for i in range(1, n+1):
        res.append((i,))
    res.append((-1,))
    return res


def generate_random_matrix_for_solution(solution: list[int], seed: int = 0) -> GF2Mat:
    col_names = generate_col_names(len(solution))
    np.random.seed(seed)
    matrix = GF2Mat(np.random.randint(2, size=(var_count*2, len(col_names))), dtype=np.int8, columns=col_names)
    for row in matrix.values:
        row[-1] = 0
        for index, elem in enumerate(row):
            if elem == 1:
                col_name = matrix.columns[index]
                if len(col_name) == 2:
                    var1 = col_name[0]-1
                    var2 = col_name[1]-1
                    if (solution[var1] == 1 and solution[var2] == 1):
                        row[-1] = row[-1] ^ 1
                elif col_name[0] > -1:
                    var1 = col_name[0]-1
                    if (solution[var1] == 1):
                        row[-1] = row[-1] ^ 1
    return matrix

#Change to tuples
def load_data_from_file(path: str) -> tuple[int, np.array]:
    var_count: int = -1
    matrix: np.array = [] 
    with open(path, encoding="utf-8") as f:
        for line in f:
            if (line.find("(n)") > -1):
                var_count =  int(line.split(":")[1].replace(" ", ""))
            if (line.find(";") > -1):
                coefs = line.split(" ")
                coefs.pop() # remove the ; at the end
                matrix.append([coef == "1" for coef in coefs])
    return(var_count, matrix)


if (__name__ == '__main__'):
    solution = [0, 1, 1, 0]
    var_count = len(solution) # should be property of the matrix itself
    degree = 2 # should be property of the matrix itself
    matrix = generate_random_matrix_for_solution(solution)
    # print(matrix.get_all_solutions())
    # print(matrix)
    # matrix = GF2Mat(matrix._xor_same_indices())
    # print(matrix)
    # print(matrix.galois_row_reduce())
    matrix = GF2Mat(matrix._xor_same_indices())
    matrix = GF2Mat(matrix.galois_row_reduce())
    matrix = GF2Mat(matrix.generate_new_rows())
    matrix = GF2Mat(matrix.galois_row_reduce())
    matrix = GF2Mat(matrix.drop_empty_rows())
    print(matrix)
    print(matrix.get_all_solutions())
