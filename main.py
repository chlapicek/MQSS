import numpy as np
import pandas
import galois
import time

class GF2Mat(pandas.DataFrame):
    degree = 2
    solution = {}
    guessed_variables = []

    def drop_zeros_only_columns(self) -> pandas.DataFrame:
        # https://stackoverflow.com/questions/21164910/how-do-i-delete-a-column-that-contains-only-zeros-in-pandas
        return self.loc[:, (self != 0).any(axis=0)]


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


    def swap_columns(self, index1: int, index2: int) -> pandas.DataFrame:
        self.loc[index1], self.loc[index2] = self.loc[index2], self.loc[index1]


    def _xor_same_indices(self) -> pandas.DataFrame:
        # TODO Verify if the column exists first
        for x in range(2, self.degree+1):
            for index in range(1, var_count+1):
                index =  (index, )
                self[index] = self[index * x] ^ self[index]
                self = self.drop(labels=index * x, axis=1)
            return self


    def _get_next_combination(self):
        var_count = len(self.guessed_variables)
        solution = np.array([0] * var_count)
        yield solution
        while solution.min() != 1:
            shift = 0
            while True:
                if solution[(var_count-1)-shift] == 1:
                    solution[(var_count-1)-shift] = 0
                    shift += 1
                    continue
                solution[(var_count-1)-shift] = 1
                break
            yield solution
        yield [-1]


    # generate with numba paralelly?
    def get_all_solutions(self) -> list[list[int]]:
        solutions = []
        for combination in self._get_next_combination(var_count):
            if self.is_valid_solution(combination):
                solutions.append(combination.copy())
        return solutions


    def generate_new_rows(self, increase_degree: bool = False) -> pandas.DataFrame:
        rows = len(self)
        for i in range(rows):
            for j in range(var_count):
                if (j+1 in self.solution.keys()):
                    continue
                self.generate_new_row(j+1, i, increase_degree)
                if len(self) >= len(self.columns)-1:
                    return self
        return self


    def generate_new_row(self, variable: int, row_index: int, increase_degree: bool = False) -> pandas.DataFrame:
        # global degree # TODO remove when not needed
        new_row = self.iloc[row_index].copy()
        for column_name, value in new_row.items(): #if columns are swapped it may not work
            if (value == 1):
                if (variable in column_name):
                    continue
                if (len(column_name) >= self.degree):
                    # if (increase_degree):
                    #     self.degree += 1
                    # else:
                        return self
                if (len(column_name) == 1 and column_name[0] == -1):
                    new_index = (variable,)
                else:
                    new_index = tuple(sorted(list((variable,) + column_name)))
                if not new_index in new_row.keys():
                    self.insert(0, new_index, 0)
                    temp = pandas.DataFrame(new_row).T
                    temp.insert(0, new_index, 1)
                    new_row = pandas.Series(temp.iloc[0], dtype=np.int8)
                else:
                    new_row[new_index] = new_row[new_index] ^ 1
                new_row[column_name] = 0
        # modification: When a new row is generated, we can directly get it to the row echelon form by adding other rows
        # This modification should probably decrease the overhead needed for generating
        # self.loc[len(self)] = new_row
        # temp = GF2Mat(pandas.DataFrame(new_row).T).get_degree_submatrix(self.degree)
        # if not temp.empty():
        #     return self
        self.add_last_row(new_row)
        return self
    

    def add_last_row(self, row: pandas.Series) -> None:
        for i in range(min(len(row), len(self.index)-1)):
            if row[i] == 1:
                # if self[self.columns.values[i]].sum() > 0:
                if self.iloc[i][i] == 1:
                    index = self[self.columns.values[i]].to_list().index(1)
                    row ^= self.iloc[index]
                    if row.sum() == 0:
                        return
        row_ones = row[row > 0]
        if row_ones.any():
            if len(row_ones) == 1 and row[-1] == 1:
                return
            self.loc[len(self)] = row
            index = row.to_list().index(1)
            self.fix_column(index)
            if (index >= len(self)-1):
                return
            for i in range(index, len(self)):
                self.iloc[i], self.iloc[len(self)-1] = self.iloc[len(self)-1].copy(), self.iloc[i].copy()
        return


    def fix_column(self, column: int | tuple) -> None:
        if type(column) is int:
            col = self.columns[column]
        else:
            col = self[column]
        for index, cell in enumerate(self[col]):
            if cell == 1 and index != len(self)-1:
                self.iloc[index] ^= self.iloc[-1]


    def drop_empty_rows(self) -> pandas.DataFrame:
        return self.loc[~(self == 0).all(axis=1)]


    def get_degree_submatrix(self, degree: int) -> pandas.DataFrame:
        column_names = list(self.columns)
        column_names = list(filter(lambda x: len(x) == degree, column_names))
        if (degree == 0):
            column_names = [(-1,)]
        elif (degree == 1):
            column_names.pop()
        return self.loc[:, self.columns.isin(column_names)]


    def get_linear_submatrix(self) -> pandas.DataFrame:
        linear = pandas.concat([self.get_degree_submatrix(1), self.get_degree_submatrix(0)], axis=1)
        rest = self[~self.isin(linear)].dropna(axis=1)
        rows = []
        for key, row in rest.iterrows():
            if row.sum() == 0:
                rows.append(key)
        return linear.T.loc[:, linear.T.columns.isin(rows)].T # can be probably done better


    def numpy_linalg_solve(self) -> pandas.DataFrame:
        coeff_matrix = pandas.DataFrame.to_numpy(self.iloc[:, 0:len(self)], dtype=np.uint8)
        ordinate_values = pandas.DataFrame.to_numpy(self.iloc[:, len(self):len(self)+1], dtype=np.uint8)
        result = np.linalg.solve(coeff_matrix, ordinate_values).astype(np.int8).T
        columns = self.columns[:-1]
        if result.shape[1] != len(self.columns)-1:
            return pandas.DataFrame([])
        return pandas.DataFrame(result, columns=columns)


    def solve(self):
        start_time = time.time()
        self = GF2Mat(self._xor_same_indices())

        copy = GF2Mat(self.copy())
        self.set_x_variables_to_zero(var_count//2)
        guessed = self.guessed_variables.copy()
        self = GF2Mat(self.galois_row_reduce())
        cycle = True
        while cycle:
            for combination in self._get_next_combination():
                if (combination[0] == -1):
                    cycle = False
                    break
                self = GF2Mat(copy.copy())
                for i in range(len(guessed)):
                    self.set_variable(guessed[i], combination[i], True)
                self = GF2Mat(self.galois_row_reduce())

                if (self.has_solution()):
                    if (self.solve_linear()):
                        cycle = False
                        break

        print(self.solution)   
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(elapsed_time)


    def solve_linear(self) -> bool: 
        linear = GF2Mat([])
        increase_degree = False
        self = GF2Mat(self.galois_row_reduce())
        self = GF2Mat(self.drop_empty_rows())
        while (len(linear.index) + len(self.solution) < var_count):
            self = GF2Mat(self.generate_new_rows(increase_degree))
            self = GF2Mat(self.drop_zeros_only_columns())
            self = GF2Mat(self.galois_row_reduce())
            self = GF2Mat(self.drop_empty_rows())
            linear = GF2Mat(self.get_linear_submatrix())
            self.degree += 1
            if self.degree > var_count:
                break
        if not linear.has_solution():
            return False
        result = linear.numpy_linalg_solve()
        if (result.size > 0):
            result_dict = result.to_dict()
            for res in result_dict:
                self.set_variable(res[0], result_dict[res][0], False)
            return True
        return False


    def set_variable(self, variable: int, value: int, guessed: bool = False)-> None:
        self.solution[variable] = value
        if guessed:
            if (not variable in self.guessed_variables):
                self.guessed_variables.append(variable)
        columns = sorted(self.columns, key=len, reverse=True)
        for column in columns:
            if variable in column:
                if value == 1:
                    new_column = list(column)
                    new_column.remove(variable)
                    if len(new_column) == 0:
                        new_column = (-1,)
                    else:
                        new_column = tuple(new_column)
                    self[new_column] ^= self[column]
                self.drop(column, axis=1, inplace=True)


    def set_x_variables_to_zero(self, x: int) -> None:
        variables = [x for x in range(1, var_count+1)]
        variables = np.random.permutation(variables)
        variables = variables[:x]
        for variable in variables:
            self.set_variable(variable, 0, True)


def generate_col_names(n: int) -> list[str]:
    res = []
    for i in range(1, n+1):
        for j in range(1, i+1):
            res.append((j, i))
    for i in range(1, n+1):
        res.append((i,))
    res.append((-1,))
    return res


def generate_random_matrix_for_solution(solution: list[int]) -> GF2Mat:
    col_names = generate_col_names(len(solution))
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
    np.random.seed(0)
    solution_key = [0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0]
    var_count = len(solution_key) # should be property of the matrix itself
    matrix = generate_random_matrix_for_solution(solution_key)
    matrix.solve()
    
