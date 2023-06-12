import numpy as np
import pandas
import galois
import re

class GF2Mat(pandas.DataFrame):
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
        if (len(solution) != var_count): # var_count should be property of G2MAT
            return []
        col_names = []
        for col_name in self.loc[0].axes[0]:
            col_names.append(re.findall("_\d+", col_name))
        solution_row = []
        for col_name in col_names:
            if (len(col_name) == 2):
                index1 = int(col_name[0].replace("_", ""))-1
                index2 = int(col_name[1].replace("_", ""))-1
                solution_row.append(solution[index1] & solution[index2])
            elif (len(col_name) == 1):
                index = int(col_name[0].replace("_", ""))-1
                solution_row.append(solution[index])
            else:
                solution_row.append(1)
        return solution_row


    def solve(self) -> dict[str, int]:
        matrix = self
        # matrix = GF2Mat(matrix.drop_zeros_only_columns())
        matrix = GF2Mat(matrix.galois_row_reduce())
        matrix = GF2Mat(matrix._xor_same_indices())

        variables: dict[str, int] = matrix._set_last_half_variables_to_zeros()
        self = matrix
        matrix = GF2Mat(matrix._calculate_zero_columns(variables))
        return


    def swap_columns(self, index1: int, index2: int) -> pandas.DataFrame:
        self.loc[index1], self.loc[index2] = self.loc[index2], self.loc[index1]




    def _xor_same_indices(self) -> pandas.DataFrame:
        # VERIFY IF THE COLUMN EXISTS FIRST
        for index in range(var_count):
            index =  "_" + str(index+1)
            self["x"+index] = self["x"+index+index] ^ self["x"+index]
            self = self.drop(labels="x"+index+index, axis=1)
        return self
        

    def _set_last_half_variables_to_zeros(self) -> dict[str, int]:
        variables = dict()
        keys = self[self.columns[-(var_count//2)-1:]].keys().array
        for key in keys:
            variables[key] = 0
        variables.pop("1")
        return variables


    def _calculate_zero_columns(self, variables: dict[str, int]) -> pandas.DataFrame:
        matrix = self
        for column in matrix.columns:
            for variable in variables:
                variable = variable.replace("x", "")
                if column.endswith(variable) or column.find(variable+"_") > -1:
                    matrix["1"] = matrix["1"] ^ matrix[column] # not sure if we need to do that
                    matrix = matrix.drop(labels=column, axis=1)
                    break
        return matrix


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
    

    def generate_new_row(self, variable: int, row_index: int) -> pandas.DataFrame:
        new_row = self.iloc[row_index].copy()
        for column, _ in new_row.items():
            new_row[column] = 0

        for column, value in self.iloc[row_index].items():
            column = str(column)
            if (value == 1):
                if (f"_{variable}" in column):
                    continue
                if (len(column.split("_")) > 2):
                    return self
                split = column.split("_")
                if len(split) == 1:
                    new_row[f"x_{variable}"] = 1
                if len(split) == 2:
                    index = split[-1]
                    if (int(index) < variable):
                        new_row[f"x_{index}_{variable}"] = 1
                    else:
                        new_row[f"x_{variable}_{index}"] = 1
                new_row[column] = 0
        self.loc[len(self)] = new_row
        return self


def generate_col_names(n: int) -> list[str]:
    res = []
    for i in range(1, n+1):
        for j in range(1, i+1):
            res.append(f"x_{j}_{i}")
    for i in range(1, n+1):
        res.append(f"x_{i}")
    res.append("1")
    return res


def generate_random_matrix_for_solution(solution: list[int]) -> GF2Mat:
    col_names = generate_col_names(len(solution))
    np.random.seed(0)
    matrix = GF2Mat(np.random.randint(2, size=(var_count*2, len(col_names))), dtype=np.int8, columns=col_names)
    for row in matrix.values:
        row[-1] = 0
        for index, elem in enumerate(row):
            if elem == 1:
                split = matrix.columns[index].split("_")
                if len(split) == 3:
                    var1 = int(split[1]) - 1
                    var2 = int(split[2]) - 1
                    if (solution[var1] == 1 and solution[var2] == 1):
                        row[-1] = row[-1] ^ 1
                elif len(split) == 2:
                    var1 = int(split[1]) - 1
                    if (solution[var1] == 1):
                        row[-1] = row[-1] ^ 1
    return matrix


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
    var_count = len(solution)
    matrix = generate_random_matrix_for_solution(solution)
    # print(matrix.get_all_solutions())
    # print(matrix)
    # matrix = GF2Mat(matrix._xor_same_indices())
    # print(matrix)
    # print(matrix.galois_row_reduce())
    matrix = GF2Mat(matrix._xor_same_indices())
    matrix = GF2Mat(matrix.galois_row_reduce())
    matrix.generate_new_row(1, 1)
    # print(matrix)
    # print(matrix.is_valid_solution(solution))
    # matrix = matrix._xor_same_indices()
    # print(matrix)
    # var_count, matrix = load_data_from_file("data/type1-n4-seed0")

    # col_names = generate_col_names(var_count)

    # test: GF2Mat = GF2Mat(matrix, columns=col_names, dtype=np.uint8)
    # test.solve()
    # print(test.is_valid_solution([0, 1, 0, 1]))
