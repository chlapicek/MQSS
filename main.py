import numpy as np
import pandas
import galois
import time
import itertools
import numba


class GF2Mat(pandas.DataFrame):
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


    def _xor_same_indices(self, drop: bool = True) -> pandas.DataFrame:
        # TODO Verify if the column exists first
        for x in range(2, degree+1):
            for index in range(1, var_count+1):
                index =  (index, )
                self[index] = self[index * x] ^ self[index]
                if drop:
                    self = self.drop(labels=index * x, axis=1)
                else:
                    self[index] = 0
            return self


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


    # def solve(self):
    #     start_time = time.time()
    #     self = GF2Mat(self._xor_same_indices())

    #     copy = GF2Mat(self.copy())
    #     self.set_x_variables_to_zero(var_count//2)
    #     guessed = self.guessed_variables.copy()
    #     self = GF2Mat(self.galois_row_reduce())
    #     cycle = True
    #     while cycle:
    #         for combination in self._get_next_combination():
    #             if (combination[0] == -1):
    #                 cycle = False
    #                 break
    #             self = GF2Mat(copy.copy())
    #             for i in range(len(guessed)):
    #                 self.set_variable(guessed[i], combination[i], True)
    #             self = GF2Mat(self.galois_row_reduce())

    #             if (self.has_solution()):
    #                 if (self.solve_linear()):
    #                     cycle = False
    #                     break

    #     print(self.solution)   
    #     end_time = time.time()
    #     elapsed_time = end_time - start_time
    #     print(elapsed_time)


    # def solve_linear(self) -> bool: 
    #     linear = GF2Mat([])
    #     generate_more = 1
    #     self = GF2Mat(self.galois_row_reduce())
    #     self = GF2Mat(self.drop_empty_rows())
    #     while (len(linear.index) + len(self.solution) < var_count):
    #         self = GF2Mat(self.generate_new_rows(generate_more))
    #         self = GF2Mat(self.drop_zeros_only_columns())
    #         # self = GF2Mat(self.galois_row_reduce())
    #         self = GF2Mat(self.drop_empty_rows())
    #         linear = GF2Mat(self.get_linear_submatrix())
    #         if (generate_more >= 3):
    #             self.degree += 1
    #             generate_more = 1
    #         generate_more += 1
    #         if self.degree > var_count:
    #             break
    #     if not linear.has_solution():
    #         return False
    #     result = linear.numpy_linalg_solve()
    #     if (result.size > 0):
    #         result_dict = result.to_dict()
    #         for res in result_dict:
    #             self.set_variable(res[0], result_dict[res][0], False)
    #         return True
    #     return False


    def set_variable(self, variable: int, value: int, guessed: bool = False, drop: bool = True)-> None:
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
                if (drop):
                    self.drop(column, axis=1, inplace=True)
                else:
                    self[column] = 0


    def set_x_variables_to_zero(self, x: int, drop: bool = True) -> None:
        variables = [x for x in range(1, var_count+1)]
        variables = np.random.permutation(variables)
        variables = variables[:x]
        for variable in variables:
            self.set_variable(variable, 0, True, drop)


    def is_in_rref(self) -> bool:
        zero_row = False
        prev_index = -1
        for i in range(len(self)):
            row = self.iloc[i]
            row_ones = row[row > 0]
            if row_ones.sum() == 0:
                zero_row = True
                continue
            elif row_ones.sum() > 0 and zero_row:
                return False
            index = row_ones.index[0]
            if self[index].sum() > 1:
                return False
            if self.columns.get_loc(index) <= prev_index:
                return False
            prev_index = self.columns.get_loc(index)
        return True


    @numba.jit
    def generate_rows(self, prev_row_count: int):
        new_index = prev_row_count
        for i in range(var_count):
            if (i+1 in self.solution.keys()):
                continue
            for j in range(prev_row_count):
                if (self.generate_row(i+1, j, new_index)):
                    new_index += 1


    @numba.jit
    def generate_row(self, variable: int, row_index: int, row_insert: int) -> bool:
        new_row = self.iloc[row_index].copy()
        for column_name, value in new_row.items(): #if columns are swapped it may not work
            if (value == 1):
                if (variable in column_name):
                    continue
                if (len(column_name) == 1 and column_name[0] == -1):
                    new_index = (variable,)
                else:
                    new_index = tuple(sorted(list((variable,) + column_name)))
                if (len(new_index) > degree):
                    return False
                new_row[new_index] = new_row[new_index] ^ 1
                new_row[column_name] = 0
        self.iloc[row_insert] = new_row
        return True


    # @numba.jit
    def test(self) -> None:
        global degree
        
        start_time = time.time()
        finished: bool = False

        self = GF2Mat(self._xor_same_indices())
        copy = GF2Mat(self.copy())
        self.set_x_variables_to_zero(var_count//2)
        guessed = self.guessed_variables.copy()
        total = pow(2, var_count//2)
        comb = 0
        # degree_rows = {}
        for combination in self._get_next_combination(var_count//2):
            if finished:
                break
            comb += 1
            print(f"{comb} out of {total}, {combination}")
            self = GF2Mat(copy.copy())
            for i in range(len(guessed)):
                self.set_variable(guessed[i], combination[i], True)
            # degree_rows[0] = range(0, 0)
            # degree_rows[1] = range(0, 0)
            # degree_rows = [0] # should probably contain ranges, doesn't work after galois_row_reduce()
            # degree_rows.append(self.shape[0])
            while True:
                degree += 1
                # degree_rows.append(self.shape[0] - sum(degree_rows))
                new_col_names = generate_all_column_names(degree, var_count, guessed, True)
                prev_row_count = self.shape[0]
                total_row_count =  ((var_count - len(guessed)) + 1) * prev_row_count
                self = GF2Mat(self.reindex(columns=new_col_names, index=range(total_row_count), fill_value=0))
                self.generate_rows(prev_row_count)
                if (self.shape[0] < self.shape[1]-1):
                    continue
                self = GF2Mat(self.galois_row_reduce())
                self = GF2Mat(self.drop_empty_rows())
                self = GF2Mat(self.drop_zeros_only_columns())
                if (self.shape[0] < self.shape[1]-1):
                    continue
                if (self.has_solution()):
                    linear = GF2Mat(self.get_linear_submatrix())
                    result = linear.numpy_linalg_solve()
                    if (len(result)):
                        test_solution = []
                        dict_result = result.to_dict()
                        for x in range(1, var_count+1):
                            if x in self.solution.keys():
                                test_solution.append(self.solution[x])
                            else:
                                test_solution.append(dict_result[(x,)][0])
                        if (copy.is_valid_solution(test_solution)):
                            finished = True
                            print(result)
                            print(self.solution)
                degree = 2
                break
                
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(elapsed_time)


    def _get_next_combination(self, var_count: int):
        solutions = itertools.product([0, 1], repeat=var_count)
        for solution in solutions:
            yield solution


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


def generate_column_names(tuple_length: int, max_number: int, ignore_vars:list[int], filter_same: bool) -> list[tuple]:
    vars: list = list(range(1, max_number+1))
    vars = np.setdiff1d(vars, ignore_vars)
    names = list(itertools.combinations_with_replacement(vars, tuple_length))
    if (filter_same):
        names = list(filter(lambda x: len(set(x)) == tuple_length, names))
    for i in range(tuple_length):
        names.sort(key=lambda x: x[i])
    return names


def generate_all_column_names(highest_tuple_length: int, max_number: int, ignore_vars:list[int]=[], filter_same=False):
    result = []
    for i in range(highest_tuple_length, 0, -1):
        result.extend(generate_column_names(i, max_number, ignore_vars, filter_same))
    result.extend([(-1,)])
    return result



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
    # solution_key = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, ] # 20
    # solution_key = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, ] # 15
    # solution_key = [0, 1, 1, 0, 1, ] # 5
    # var_count = len(solution_key) # should be property of the matrix itself
    degree = 2
    # matrix = generate_random_matrix_for_solution(solution_key)
    var_count, matrix = load_data_from_file("C:\\Users\\vojts\\Documents\\school\\bakalarka\\MQSS\\data\\challenge-1-55-0\\challenge-1-55-0")
    col_names = generate_all_column_names(degree, var_count)
    matrix = GF2Mat(matrix, columns=col_names, dtype=np.int0)
    # matrix.solve()
    matrix.test()


# Zprovoznit řešení, leč i pomalé. OK
# Správně počítat potřebný počet řádků
# Začít Zahazovat proměnné, reindexovat OK?
# Najít popis algoritmu pro underdetermined (méně rovnic jak proměnných)
# Zapojit Pollard-Rho