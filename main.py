import numpy as np
import pandas
import galois
# import time
import itertools
import re
# import numba
import cProfile
import pstats
import multiprocessing.pool
from collections import defaultdict
from copy import deepcopy
from z3 import *
from tqdm import tqdm

class GF2Mat(pandas.DataFrame):
    most_common: dict[str, int] = {}
    vars_needed: int = -1

    def drop_zeros_only_columns(self) -> pandas.DataFrame:
        # https://stackoverflow.com/questions/21164910/how-do-i-delete-a-column-that-contains-only-zeros-in-pandas
        return self.loc[:, (self != 0).any(axis=0)]


    def galois_row_reduce(self) -> pandas.DataFrame:
        np_arr = pandas.DataFrame.to_numpy(self, dtype=np.uint8)
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


    def num_of_satisfied_equations(self, solution: list[int]) -> int:
        solution_row = self._get_solution_row(solution)
        if (len(solution_row) == 0):
            return 0 # throw ex?
        return sum((self & solution_row).sum(axis=1) % 2 == 0)


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


    def _xor_same_indices(self, drop: bool = True):
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
        if (self.iloc[:, :-1] == 0).all(axis=1).any():
            return pandas.DataFrame([])
        coeff_matrix = pandas.DataFrame.to_numpy(self.iloc[:, 0:len(self)], dtype=np.uint8)
        ordinate_values = pandas.DataFrame.to_numpy(self.iloc[:, len(self):len(self)+1], dtype=np.uint8)
        result = np.linalg.solve(coeff_matrix, ordinate_values).astype(np.int8).T
        columns = self.columns[:-1]
        if result.shape[1] != len(self.columns)-1:
            return pandas.DataFrame([])
        return pandas.DataFrame(result, columns=columns)


    def set_variable(self, variable: int, value: bool, drop: bool = True)-> None:
        columns = sorted(self.columns, key=len, reverse=True)
        for column in columns:
            if variable in column:
                if value:
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


    def transform_to_z3(self, row_indices: list[int], ctx: Context) -> Solver:
        solver = Solver(ctx=ctx)
        colNameToVar = {}
        counter = 1

        for column in self.columns:
            if len(column) > 1:
                colNameToVar[column] = Bool(f"y{counter}", ctx)
                counter += 1
            elif column == (-1,):
                continue
            else:
                colNameToVar[column] = Bool(f"x{column[0]}", ctx)

        for column in self.columns:
            if len(column) > 1:
                solver.add(And( Or(colNameToVar[column], Not(colNameToVar[(column[0],)]), Not(colNameToVar[(column[1],)])),
                                Or(colNameToVar[(column[0],)], Not(colNameToVar[column])),
                                Or(colNameToVar[(column[1],)], Not(colNameToVar[column]))))

        for index in row_indices:
            is_odd = False
            vars = []
            for col_name, value in self.iloc[index].items():
                if col_name == (-1, ) and value == 1:
                    is_odd = True
                elif value == 1:
                    vars.append(colNameToVar[col_name])
            solver.add(Sum([If(b, 1, 0) for b in vars]) % 2 == is_odd)

        return solver

    # add threadpools for init?
    def init_solvers(self, group_size: int, solver_count: int) -> tuple[list[Solver], dict[Solver, Context]]:
        solvers = []
        contexts: dict[Solver, Context] = {}
        
        for _ in range(solver_count):
            rows = list(self.sample(n=group_size).T.columns)
            solver_ctx = Context()
            solver = self.transform_to_z3(rows, solver_ctx)
            if solver.check() != sat:
                return ([], {})
            contexts[solver] = solver_ctx
            solvers.append(solver)
        return solvers, contexts
    
    
    def set_var(self, var_name: str, var: bool, assumptions: list, determined_vars: dict[str, bool]) -> None:
        determined_vars[var_name] = var
        assumptions.append(Bool(var_name) if var else Not(Bool(var_name)))


    def set_vars(self, combination: tuple[bool, ...], assumptions: list, determined_vars: dict[str, bool], offset: int = 0) -> None:
        most_common_keys = list(self.most_common.keys())
        for index, var in enumerate(combination):
            self.set_var(most_common_keys[index], var, assumptions, determined_vars)


    def determine_additional_vars(self, stats: dict[str, int], assumptions: list, determined_vars: dict[str, bool], threshold: int=0) -> None:
        max_stat = -1
        max_count = 0
        min_stat = np.inf
        min_count = 0
        
        reversed_stats = find_keys_with_same_values(stats)
        for key in reversed_stats:
            arr = reversed_stats[key]
            for i in range(len(arr)-1):
                assumptions.append(Bool(arr[i]) == Bool(arr[i+1]))


        for key in stats.keys():
            if stats[key] == 0 and not (key in determined_vars):
                determined_vars[key] = False
                assumptions.append(Not(Bool(key)))
            if stats[key] >= max_stat-threshold and not (key in determined_vars):
                if stats[key] == max_stat:
                    max_count += 1
                else:
                    max_count = 1
                max_stat = stats[key]
            if stats[key] <= min_stat+threshold and not (key in determined_vars):
                if stats[key] == min_stat:
                    min_count += 1
                else:
                    min_count = 1
                    min_stat = stats[key]
        
        if max_count > min_count:
            for key in stats.keys():
                if stats[key] >= max_stat-threshold and not (key in determined_vars):
                    determined_vars[key] = True
                    assumptions.append(Bool(key))
        elif min_count > 0:
            for key in stats.keys():
                if stats[key] <= min_stat+threshold and not (key in determined_vars):
                    determined_vars[key] = False
                    assumptions.append(Not(Bool(key)))
    
    
    def solve_linear(self, determined_vars: dict[str, bool]) -> pandas.DataFrame:
        matrix = GF2Mat(self.copy())
        for key, value in determined_vars.items():
            var = int(re.search(r'\d+', key).group())
            matrix.set_variable(var, value)
        linear = GF2Mat(matrix.get_linear_submatrix())
        linear = GF2Mat(linear.galois_row_reduce())
        linear = GF2Mat(linear.drop_empty_rows())
        if linear.shape[0] < linear.shape[1] - 1:
            return pandas.DataFrame()
        if linear.shape[0] == linear.shape[1] - 1:
            return linear.numpy_linalg_solve()
        
        linear = GF2Mat(linear[:-(linear.shape[0] - (linear.shape[1]-1))])
        return linear.numpy_linalg_solve()
        
        


    def solve_combination(self, combination: tuple[bool, ...], solvers: list[Solver], contexts: dict[Solver, Context], offset: int=0) -> list[int]:
        determined_vars: dict[str, bool] = {}
        assumptions = []
        stats: dict[str, int] = {}
        self.set_vars(combination, assumptions, determined_vars, offset)

        vars_needed = self.vars_needed

        while len(determined_vars) < vars_needed:
            self.determine_additional_vars(stats, assumptions, determined_vars, var_count*2)
            
            assumptions_ctx = {}
            for solver in solvers:
                assumptions_ctx[solver] = [deepcopy(assumption).translate(contexts[solver]) for assumption in assumptions]

            with multiprocessing.pool.ThreadPool(len(solvers)) as pool:
                all_results = pool.starmap(self.check_solver, [(solver, assumptions_ctx[solver]) for solver in solvers])
            if not all(all_results):
                break
                    
            result = []
            for i in range(1, var_count+1):
                result.append(1 if f"x{i}" in determined_vars and determined_vars[f"x{i}"] else 0)
            is_result = self.is_valid_solution(result)
            if is_result:
                print(result)
                return result
            
            stats = {f'x{i}': 0 for i in range(1, var_count+1)}
            
            with multiprocessing.pool.ThreadPool(len(solvers)) as pool:
                all_stats = pool.map(self.get_stats, [solver for solver in solvers])

            for stat, is_res in all_stats:
                if is_res:
                    result = list({key: stat[key] for key in sorted(stat, key=lambda x: int(x[1:]))}.values())
                    print(result)
                    return result
                for key in stat.keys():
                    stats[key] += stat[key]
        
        res = self.solve_linear(determined_vars)
        
        if len(res):
            for key, value in res.to_dict().items():
                determined_vars[f"x{key[0]}"] = value[0] == 1

            result = []
            for i in range(1, var_count+1):
                result.append(1 if f"x{i}" in determined_vars and determined_vars[f"x{i}"] else 0)
            is_result = self.is_valid_solution(result)
            if is_result:
                print(result)
                return result
        return []


    def find_solution(self, group_size: int = 10, solver_count: int = 10, start_len: int = 1):
        solvers, contexts = self.init_solvers(group_size, solver_count)
                
        self.set_most_common()
        self.set_vars_needed()

        for i in range(start_len, var_count+1):
            combinations = itertools.product([False, True], repeat=i)
            for combination in tqdm(combinations, total=pow(2, i)):
                res = self.solve_combination(combination, solvers, contexts)
                if len(res):
                    return res
               

    def check_solver(self, solver: Solver, assumptions: list) -> bool:
        return solver.check(assumptions) == sat


    def get_stats(self, solver: Solver) -> tuple[dict[str, int], bool]:
        stats: dict[str, int] = {f'x{i}': 0 for i in range(1, var_count+1)}
        model = solver.model()
        for res in model.decls():
            if str(res).startswith('x'):
                if model[res]:
                    stats[str(res)] += 1
        result = list({key: stats[key] for key in sorted(stats, key=lambda x: int(x[1:]))}.values())
            
        factor = self.num_of_satisfied_equations(result)
        
        if factor >= self.shape[0] - 1:
            return stats, True
        
        modified_stats = {key: value * factor for key, value in stats.items()}

        return modified_stats, False


    def set_most_common(self) -> None:
        most_common: dict[str, int] = {f'x{i}': 0 for i in range(1, var_count+1)}
        for column in self.columns:
            if column == (-1,):
                continue
            col_sum = self[column].sum()
            for var in column:
                most_common[f'x{var}'] += col_sum

        self.most_common = dict(sorted(most_common.items(), key=lambda item: item[1], reverse=True))


    def set_vars_needed(self) -> None:
        row_count = self.shape[0]
        col_count = self.shape[1]
        x = Int('x')

        opt = Optimize()

        opt.add(x <= var_count, x >= 0)
        opt.add(col_count - (((var_count - x + 1) * (var_count + x)) / 2) <= row_count - x)

        opt.maximize(x)

        opt.check()

        self.vars_needed = var_count - opt.model()[x].as_long()


def find_keys_with_same_values(my_dict: dict) -> dict:
    inverted_dict: dict = defaultdict(list)
    for key, value in my_dict.items():
        inverted_dict[value].append(key)
    
    return {value: keys for value, keys in inverted_dict.items() if len(keys) > 1}


def generate_col_names(n: int) -> list[tuple[int, int|None]]:
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
                    var1 = int(col_name[0])-1
                    var2 = int(col_name[1])-1
                    if (solution[var1] == 1 and solution[var2] == 1):
                        row[-1] = row[-1] ^ 1
                elif int(col_name[0]) > -1:
                    var1 = int(col_name[0])-1
                    if (solution[var1] == 1):
                        row[-1] = row[-1] ^ 1
    return matrix


def generate_column_names(tuple_length: int, max_number: int, ignore_vars:list[int], filter_same: bool) -> list[tuple]:
    vars: list = list(range(1, max_number+1))
    modified_vars = np.setdiff1d(vars, ignore_vars)
    names = list(itertools.combinations_with_replacement(modified_vars, tuple_length))
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



def load_data_from_file(path: str) -> tuple[int, list]:
    var_count: int = -1
    matrix = [] 
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

    solution_key = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, ] # 30
    # solution_key = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, ] # 20
    # solution_key = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, ] # 15
    # solution_key = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1, ] # 10
    # solution_key = [0, 1, 1, 0, 1, ] # 5
    var_count = len(solution_key) # should be property of the matrix itself
    degree = 2
    matrix = generate_random_matrix_for_solution(solution_key)
    # var_count, matrix = load_data_from_file("C:\\Users\\vojts\\Documents\\school\\bakalarka\\MQSS\\data\\challenge-1-55-0\\challenge-1-55-0")
    # col_names = generate_all_column_names(degree, var_count)
    # matrix = GF2Mat(matrix, columns=col_names, dtype=np.int0)


    matrix = GF2Mat(matrix._xor_same_indices())
    matrix = GF2Mat(matrix.galois_row_reduce())

    cProfile.run("matrix.find_solution(10, 10, 10)", "test_profile")

    # cProfile.run("matrix.test()", "test_profile")
    stats = pstats.Stats("test_profile")
    stats.sort_stats('cumulative')
    stats.print_stats(20)
