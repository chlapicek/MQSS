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
        # matrix = self
        # matrix = GF2Mat(matrix.drop_zeros_only_columns())
        # matrix = GF2Mat(matrix.galois_row_reduce())
        # matrix = GF2Mat(matrix._xor_same_indices())

        # variables: dict[str, int] = matrix._set_last_half_variables_to_zeros()
        # matrix = GF2Mat(matrix._calculate_zero_columns(variables))
        #
        raise NotImplementedError

    def _xor_same_indices(self) -> pandas.DataFrame:
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


def generate_col_names(n: int) -> list[str]:
    res = []
    for i in range(1, n+1):
        for j in range(1, i+1):
            res.append(f"x_{j}_{i}")
    for i in range(1, n+1):
        res.append(f"x_{i}")
    res.append("1")
    return res


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
    var_count, matrix = load_data_from_file("data/type1-n4-seed0")

    col_names = generate_col_names(var_count)

    test: GF2Mat = GF2Mat(matrix, columns=col_names, dtype=np.uint8)
    test.solve()
    print(test.is_valid_solution([1, 0, 1, 0]))
