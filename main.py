import numpy as np
import pandas
import galois
# import time
import itertools
from functools import reduce
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
    """Main class extending pandas.DataFrame that contains the whole logic for all of the computation.
    The Matrix/DataFrame is a representation of a Galois Field(2).
    """
    most_common: dict[str, int] = {}
    vars_needed: int = -1

    def drop_empty_columns(self) -> pandas.DataFrame:
        """Drops columns containing only zeros.
        
        Does NOT modify self.

        Returns:
            pandas.DataFrame: new Dataframe without zero columns
        """
        return self.loc[:, (self != 0).any(axis=0)]
    
    

    def drop_empty_rows(self) -> pandas.DataFrame:
        """Drops rows containing only zeros.

        Does NOT modify self.
        
        Returns:
            pandas.DataFrame: new Dataframe without zero rows
        """
        return self.loc[~(self == 0).all(axis=1)]


    def galois_row_reduce(self) -> pandas.DataFrame:
        """Uses galois library to generate a DataFrame that is in reduced row echelon form.
        
        Does NOT modify self 

        Returns:
            pandas.DataFrame: new Dataframe in reduced row echelon form
        """
        np_arr = pandas.DataFrame.to_numpy(self, dtype=np.uint8)
        gf2 = galois.GF2(np_arr, dtype=np.uint8)
        gf2 = gf2.row_reduce()
        return pandas.DataFrame(gf2, columns=self.columns)


    def has_solution(self) -> bool:
        """Checks whether or not *self* contains a row that has 1 only in the last column  

        Returns:
            bool
        """
        for row in self.values:
            if row[-1] == 1:
                if sum(row) - row[-1] == 0:
                    return False        
        return True


    def is_valid_solution(self, solution: list[int]) -> bool:
        """Checks if a given *solution* is solution for the whole system

        Args:
            solution (list[int]): sorted list of keys (i. e. [x_1, x_2, ..., x_n])

        Returns:
            bool
        """
        solution_row = self._get_solution_row(solution)
        if (len(solution_row) == 0):
            return False
        return ((self & solution_row).sum(axis=1) % 2 == 0).min()


    def num_of_satisfied_equations(self, solution: list[int]) -> int:
        """Calculates the number of satisfied rows for a given *solution*

        Args:
            solution (list[int]): sorted list of keys (i. e. [x_1, x_2, ..., x_n])

        Returns:
            int: number of satisfied rows for a given solution
        """
        solution_row = self._get_solution_row(solution)
        if (len(solution_row) == 0):
            return 0
        return sum((self & solution_row).sum(axis=1) % 2 == 0)


    def _get_solution_row(self, solution: list[int]) -> list[int]:
        """Generates a special row from *solution* for easier checks of a whole matrix/DataFrame

        Args:
            solution (list[int]): sorted list of keys (i. e. [x_1, x_2, ..., x_n])

        Returns:
            list[int]: solution row
        """
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


    def _xor_same_indices(self, drop: bool = True) -> pandas.DataFrame:
        """Xors columns that are composed from same indices. It can be done because the matrix is based on GF(2).
        For example: (1, 1) and (1, )
        
        Modifies self

        Args:
            drop (bool, optional): Determines if the empty column should be dropped. Defaults to True.

        Returns:
            pandas.DataFrame: modified self
        """
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


    def get_degree_submatrix(self, degree: int) -> pandas.DataFrame:
        """Creates a submatrix containing only columns with a given tuple length (degree)

        Args:
            degree (int): degree/tuple length
        Returns:
            pandas.DataFrame: new DataFrame containing only specified columns
        """
        column_names = list(self.columns)
        column_names = list(filter(lambda x: len(x) == degree, column_names))
        if (degree == 0):
            column_names = [(-1,)]
        elif (degree == 1):
            column_names.pop()
        return self.loc[:, self.columns.isin(column_names)]


    def get_linear_submatrix(self) -> pandas.DataFrame:
        """Creates a linear submatrix using self.get_degree_submatrix (degree 1 and 2)

        Returns:
            pandas.DataFrame: new DataFrame containing only linear submatrix
        """
        linear = pandas.concat([self.get_degree_submatrix(1), self.get_degree_submatrix(0)], axis=1)
        rest = self[~self.isin(linear)].dropna(axis=1)
        rows = []
        for key, row in rest.iterrows():
            if row.sum() == 0:
                rows.append(key)
        return linear.T.loc[:, linear.T.columns.isin(rows)].T # can be probably done better


    def numpy_linalg_solve(self) -> pandas.DataFrame:
        """Gets a linear submatrix and tries to solve it using numpy.linalg

        Returns:
            pandas.DataFrame: If there is a solution, it returns it in a DataFrame. Otherwise returns empty DataFrame.
        """
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
        """Sets a variable to a given value. Setting a variable means XORing columns
        
        Modifies self
        
        Examples:
            if value is False, all of the columns containing variable in a column name are set to zero\n
            If value is True, all of the columns containing variable are XORed with a column with a smaller degree.\n
            (i. e. for variable=1 (1, 2) XOR (2,), (1, 3) XOR (3,), (1,) XOR (-1,))\n

        Args:
            variable (int): Int representing the var (i. e. x_1 = 1, ..., x_n = n) 
            value (bool): What value should we set for the variable
            drop (bool, optional): Whether or not columns should be dropped. Defaults to True.
        """
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
        """Creates a solver from given row indices from self with a given context
        The rules for the transformation/creation are the following:\n
            1. For every column name of degree 1 create a Bool(x_{index})\n
            2. For every column name of higher degree create And statement containing all the Bools\n
            3. For each given row create an assertion constructed from XORs of each 1 in a row\n

        Args:
            row_indices (list[int]): list of row indices from self 
            ctx (Context): Context related to the solver and Vars

        Returns:
            Solver: z3.Solver
        """
        solver = Solver(ctx=ctx)
        colNameToVar = {}

        col_names = sorted(self.columns, key=len)

        for column in col_names:
            if len(column) == 1 and column != (-1,):
                colNameToVar[column] = Bool(f"x{column[0]}", ctx)
            elif len(column) > 1:
                colNameToVar[column] = And(list(map(lambda index: colNameToVar[(index,)], column)))

        for index in row_indices:
            is_odd = False
            vars = []
            for col_name, value in self.iloc[index].items():
                if col_name == (-1, ) and value == 1:
                    is_odd = True
                elif value == 1:
                    vars.append(colNameToVar[col_name])
            xor_chain = reduce(lambda x, y: Xor(x, y), vars)
            solver.add(xor_chain == is_odd)

        return solver


    def init_solvers(self, group_size: int, solver_count: int) -> tuple[list[Solver], dict[Solver, Context]]:
        """Creates Solvers with randomly selected rows

        Args:
            group_size (int): number of rows that should be in every solver
            solver_count (int): number of solver to be created

        Returns:
            tuple[list[Solver], dict[Solver, Context]]: Returns also contexts for a possibility to further modify the Solvers
        """
        solvers = []
        contexts: dict[Solver, Context] = {}
        
        for _ in range(solver_count):
            rows = list(self.sample(n=group_size).T.columns)
            solver_ctx = Context()
            solver = self.transform_to_z3(rows, solver_ctx)
            contexts[solver] = solver_ctx
            solvers.append(solver)
        return solvers, contexts
    
    
    def set_var(self, var_name: str, value: bool, assumptions: list, determined_vars: dict[str, bool]) -> None:
        """Modifies assumptions and determined_vars for Solvers
        """
        determined_vars[var_name] = value
        assumptions.append(Bool(var_name) if value else Not(Bool(var_name)))


    def set_vars(self, combination: tuple[bool, ...], assumptions: list, determined_vars: dict[str, bool]) -> None:
        """Modifies assumptions and determined_vars for Solvers multiple times.
            The vars to set are selected based on the frequency of 1 in their columns.         
        """
        most_common_keys = list(self.most_common.keys())
        for index, var in enumerate(combination):
            self.set_var(most_common_keys[index], var, assumptions, determined_vars)


    def determine_additional_vars(self, stats: dict[str, int], assumptions: list, determined_vars: dict[str, bool], threshold: int=0) -> None:
        """Determines additional vars for Solvers based on given stats.

        Modifies assumptions and determined vars.

        Args:
            stats (dict[str, int]): Statistics for every var and number of occurences of 1 in the corresponding columns
            assumptions (list): List of z3.Bools
            determined_vars (dict[str, bool]): Dictionary of already determined vars (represented as "x_{index}") and its values
            threshold (int, optional): Shifts the limit to determine additional var. The higher the limit, the more variables will be determined in a single call. Defaults to 0.
        """
        max_stat = -1
        max_count = 0
        min_stat = np.inf
        min_count = 0
        
        # reversed_stats = find_keys_with_same_values(stats)
        # for key in reversed_stats:
        #     if key > (self.shape[0]) and key < max(reversed_stats) - (self.shape[0]):
        #         continue
        #     arr = reversed_stats[key]
        #     for i in range(len(arr)-1):
        #         if arr[i] in determined_vars:
        #             assumptions.append(Bool(arr[i+1]) if determined_vars[arr[i]] else Not(Bool(arr[i+1])))
        #             determined_vars[arr[i+1]] = determined_vars[arr[i]]
        #         elif arr[i+1] in determined_vars:
        #             assumptions.append(Bool(arr[i]) if determined_vars[arr[i+1]] else Not(Bool(arr[i])))
        #             determined_vars[arr[i]] = determined_vars[arr[i+1]]
        #         else:
        #             expr = Bool(arr[i]) == Bool(arr[i+1])
        #             if expr not in assumptions:
        #                 assumptions.append(expr)


        for key in stats.keys():
            if stats[key] == 0 and not (key in determined_vars):
                determined_vars[key] = False
                assumptions.append(Not(Bool(key)))
            if stats[key] >= max_stat-threshold and (not (key in determined_vars) or not determined_vars[key]):
                if stats[key] == max_stat:
                    max_count += 1
                else:
                    max_count = 1
                max_stat = stats[key]
            if stats[key] <= min_stat+threshold and (not (key in determined_vars) or determined_vars[key]):
                if stats[key] == min_stat:
                    min_count += 1
                else:
                    min_count = 1
                    min_stat = stats[key]
        
        if max_count > min_count:
            for key in stats.keys():
                if stats[key] >= max_stat-threshold and (not (key in determined_vars) or not determined_vars[key]):
                    determined_vars[key] = True
                    var = Bool(key)
                    if var in assumptions:
                        assumptions[assumptions.index(var)] = Not(var)
                    else:
                        assumptions.append(var)
        elif min_count > 0:
            for key in stats.keys():
                if stats[key] <= min_stat+threshold and (not (key in determined_vars) or determined_vars[key]):
                    determined_vars[key] = False
                    var = Bool(key)
                    if Not(var) in assumptions:
                        assumptions[assumptions.index(Not(var))] = var
                    else:
                        assumptions.append(Not(var))
    
    
    def solve_linear(self, determined_vars: dict[str, bool]) -> pandas.DataFrame:
        """Solves a linear matrix using numpy lingalg module. The linear matrix is created using setting a variable for the GF2Mat (self._set_variable).
        It returns empty DataFrame, if there is no solution or the linear matrix has more columns than rows.
        If the linear matrix has more rows than cols, it takes just the first rows to find a solution.
        
        (The last point means it can return a solution, that is not valid for the whole matrix,
        therefore after a solve_linear call there should be a check if the solution satisfies the whole matrix)

        Args:
            determined_vars (dict[str, bool]): Dictionary of already determined vars (represented as "x_{index}") and its values

        Returns:
            pandas.DataFrame: solution for the linear matrix, or empty DataFrame
        """
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
        
        


    def solve_combination(self, combination: tuple[bool, ...], solvers: list[Solver], contexts: dict[Solver, Context], threshold: int = 0) -> list[int]:
        """Tries to find a solution starting with a given bool combination.

        Args:
            combination (tuple[bool, ...]): bools to set in the beginning
            solvers (list[Solver]): list of used Solvers
            contexts (dict[Solver, Context]): Context related to the Solvers
            threshold (int, optional): Used in self.determine_additional_vars. Defaults to 0.

        Returns:
            list[int]: Returns the result, or an empty array.

            Example of a result:
             x1 x2 x3 x4 x5
            [1, 0, 0, 1, 1]
        """
        determined_vars: dict[str, bool] = {}
        assumptions = []
        stats: dict[str, int] = {}
        self.set_vars(combination, assumptions, determined_vars)

        vars_needed = self.vars_needed

        while len(determined_vars) < vars_needed:
            self.determine_additional_vars(stats, assumptions, determined_vars, threshold)
            
            assumptions_ctx = {}
            for solver in solvers:
                assumptions_ctx[solver] = [deepcopy(assumption).translate(contexts[solver]) for assumption in assumptions]

            with multiprocessing.pool.ThreadPool(len(solvers)) as pool:
                all_results = pool.starmap(self.check_solver, [(solver, assumptions_ctx[solver], contexts[solver]) for solver in solvers])
            if not all(all_results):
                return []

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


    def find_solution(self, group_size: int = 10, solver_count: int = 10, start_len: int = 1, skip: int = 0, threshold: int = 0) -> list[int]:
        """Main function to use for finding the result for a given matrix/Dataframe.

        Args:
            group_size (int, optional): Number of rows for each solver. Defaults to 10.
            solver_count (int, optional): Number of solvers. Defaults to 10.
            start_len (int, optional): length of binary combination to start with. Defaults to 1.
            skip (int, optional): Number of combinations to skip in the beginning. Defaults to 0.
            threshold (int, optional): Used in self.determine_additional_vars. Defaults to 0.

        Returns:
            list[int]: Returns the result, or an empty array.

            Example of a result:
             x1 x2 x3 x4 x5
            [1, 0, 0, 1, 1]
        """
        solvers, contexts = self.init_solvers(group_size, solver_count)
                
        self.set_most_common()
        self.set_vars_needed()
        
        counter = 0

        for i in range(start_len, var_count+1):
            combinations = itertools.product([False, True], repeat=i)
            for combination in tqdm(combinations, total=pow(2, i)):
                if skip > 0:
                    skip -= 1
                    continue
                res = self.solve_combination(combination, solvers, contexts, threshold)
                if len(res):
                    return res
                # it's good performance-wise to reset the solvers from time to time. Can probably be done with some Solver settings
                if counter >= 50:
                    solvers, contexts = self.init_solvers(group_size, solver_count)
                    counter = 0
                counter += 1
        return []


    def check_solver(self, solver: Solver, assumptions: list, ctx: Context) -> bool:
        """Checks solver if it is satisfiable with given assumptions.
        
        Modifies assumptions.  

        Returns:
            bool: Return whether or not solver with given assumptions has a solution
        """
        if solver.check(assumptions) != sat:
            return False
        
        lowest = sum(set(self.get_stats(solver)[0].values()))

        if lowest == 0:
            return True

        stats: dict[str, int] = {f'x{i}': 0 for i in range(1, var_count+1)}
        model = solver.model()
        for res in model.decls():
            if str(res).startswith('x'):
                if model[res]:
                    stats[str(res)] += 1
        result = list({key: stats[key] for key in sorted(stats, key=lambda x: int(x[1:]))}.values())
            
        lowest = self.num_of_satisfied_equations(result)
        
        for key, value in stats.items():
            stats_copy = stats.copy()
            if value == 1:
                stats_copy[key] = 0
            else:
                stats_copy[key] = 1
            
            res = list({stat_key: stats_copy[stat_key] for stat_key in sorted(stats_copy, key=lambda x: int(x[1:]))}.values())
            current = self.num_of_satisfied_equations(res)
            if current > lowest:
                lowest = current
                stats[key] = (value + 1) % 2
                var = Bool(key, ctx)
                not_var = Not(var, ctx)
                if var in assumptions:
                    index = assumptions.index(var)
                    assumptions[index] = (var if stats[key] == 1 else not_var)
                elif not_var in assumptions:
                    index = assumptions.index(not_var)
                    assumptions[index] = (var if stats[key] == 1 else not_var)
                else:
                    assumptions.append(var if stats[key] == 1 else not_var)
        return solver.check(assumptions) == sat


    def get_stats(self, solver: Solver) -> tuple[dict[str, int], bool]:
        """Calculates statistics for a solver. The statistics are a number of satisfied rows for a solver solution.
        
        solver.check had to be called before and it had to return sat.

        Returns:
            tuple[dict[str, int], bool]: The first element in a tuple represents a dictionary with variable as a key and number of satisfied rows as a value,
                and the second element is a bool representing whether or not it is a solution for a whole matrix  

        Result Example:
        ({'x1': 8, 'x2': 0, 'x3': 8, 'x4': 0, 'x5': 8}, False)
        ({'x1': 1, 'x2': 0, 'x3': 0, 'x4': 1, 'x5': 1}, True)
        """
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
        """Finds the most common variables (based on amount of 1 in their columns.
        """
        most_common: dict[str, int] = {f'x{i}': 0 for i in range(1, var_count+1)}
        for column in self.columns:
            if column == (-1,):
                continue
            col_sum = self[column].sum()
            for var in column:
                most_common[f'x{var}'] += col_sum

        self.most_common = dict(sorted(most_common.items(), key=lambda item: item[1], reverse=True))


    def set_vars_needed(self) -> None:
        """Determines how many vars have to be determined in order to use linalg solve.
        """
        row_count = self.shape[0]
        col_count = self.shape[1]
        x = Int('x')

        opt = Optimize()

        opt.add(x <= var_count, x >= 0)
        opt.add(col_count - (((var_count - x + 1) * (var_count + x)) / 2) <= row_count - x)

        opt.maximize(x)

        opt.check()

        self.vars_needed = var_count - opt.model()[x].as_long()


def find_keys_with_same_values(_dict: dict) -> dict:
    """Reverses a dictionary and keeps only those, which have more than two values (prev keys) for a key (prev value)

    Returns:
        dict: {prev_value: list[prev_keys]}
    """
    inverted_dict: dict = defaultdict(list)
    for key, value in _dict.items():
        inverted_dict[value].append(key)

    return {value: keys for value, keys in inverted_dict.items() if len(keys) > 1}


def generate_random_matrix_for_solution(solution: list[int]) -> GF2Mat:
    col_names = generate_all_column_names(2, len(solution), [], False)
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

    # solution_key = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, ] # 40
    # solution_key = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, ] # 30
    solution_key = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, ] # 20
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

    group_size = 10
    solver_count = math.floor((matrix.shape[0] // group_size) * 1.5)
    start_len = var_count // 2
    skip = 0
    threshold = 0

    cProfile.run("matrix.find_solution(group_size, solver_count, start_len, skip, threshold)", "test_profile")

    stats = pstats.Stats("test_profile")
    stats.sort_stats('cumulative')
    stats.print_stats(20)
