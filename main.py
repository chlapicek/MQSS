import cProfile
import itertools
import numpy as np
import math
import pstats
from pathlib import Path
from GF2Mat import GF2Mat


def generate_random_matrix_for_solution(solution: list[int]) -> GF2Mat:
    col_names = generate_all_column_names(2, len(solution), [], False)
    matrix = GF2Mat(degree, var_count, {}, -1, np.random.randint(2, size=(var_count*2, len(col_names))), dtype=np.int8, columns=col_names)
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
    # The GF2Mat.find_solution(...) should be the entry point for all the calculations.
    # The best values for parameters for this function are stil in question, because they impact the speed of exploring and the probability of finding the solution

    # I recommend before calling the GF2Mat.find_solution(...) to modify the given matrix with GF2Mat.xor_same_indices() and GF2Mat.galois_row_reduce()
    # to decrease the number of 1's in the matrix so the z3.Solvers have to do less computation.
    
    np.random.seed(0)
    ## Uncomment this piece of code to go through all the Fukuoka-MQ challenges
    # data_dir = Path("data")
    # for challenge_dir in data_dir.iterdir():
    #     if challenge_dir.is_dir():
    #         challenge_file = challenge_dir / f"{challenge_dir.name}"
            
    #         degree = 2
    #         var_count, matrix = load_data_from_file(challenge_file)
    #         col_names = generate_all_column_names(degree, var_count)
    #         matrix = GF2Mat(degree, var_count, {}, -1, matrix, columns=col_names, dtype=np.int0)
    #         matrix = GF2Mat(matrix.degree, matrix.var_count, matrix.most_common, matrix.vars_needed, matrix.xor_same_indices())
    #         matrix = GF2Mat(matrix.degree, matrix.var_count, matrix.most_common, matrix.vars_needed, matrix.galois_row_reduce())
            
    #         group_size = 10
    #         solver_count = math.floor((matrix.shape[0] // group_size))
    #         start_len = var_count // 2
    #         skip = 0
    #         threshold = var_count * 2
            
    #         #The find.solution method should probably also return the record of time taken and how much search space was explored 
    #         result = matrix.find_solution(group_size, solver_count, start_len, skip, threshold)
            
    #         result_file = challenge_dir / "result"
    #         with open(result_file, 'w') as file:
    #             file.write(str(result))


    # Or this one for a single run with custom challenge
    solution_key = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, ] # 30
    var_count = len(solution_key)
    degree = 2
    matrix = generate_random_matrix_for_solution(solution_key)

    matrix = GF2Mat(matrix.degree, matrix.var_count, matrix.most_common, matrix.vars_needed, matrix.xor_same_indices())
    matrix = GF2Mat(matrix.degree, matrix.var_count, matrix.most_common, matrix.vars_needed, matrix.galois_row_reduce())

    group_size = 10
    solver_count = math.floor((matrix.shape[0] // group_size))
    start_len = var_count // 2
    skip = 0
    threshold = var_count * 2

    cProfile.run("matrix.find_solution(group_size, solver_count, start_len, skip, threshold)", "test_profile")

    stats = pstats.Stats("test_profile")
    stats.sort_stats('cumulative')
    stats.print_stats(20)
 