import numpy as np
import sys


def list_to_int(bitstring):
    return sum(1 << i for i, bit in enumerate(bitstring) if bit)


def int_to_set_bits(value):
    index = 0
    result = []
    while value:
        if value & 1:
            result.append(index)
        index += 1
        value >>= 1
    return np.array(result)


def evaluate(nk_table, K, solution):
    assert(nk_table.shape[0] == solution.shape[0])
    assert(nk_table.shape[1] == (2 << K))
    circular = np.concatenate([solution, solution])
    score = 0
    for i in range(solution.shape[0]):
        index = list_to_int(circular[i:i + K + 1])
        score += nk_table[i][index]
    return score


def brute_force(nk_table, K):
    assert(nk_table.shape[1] == (2 << K))
    N = nk_table.shape[0]
    # Set it arbitrarily bad
    best_score = -sys.maxint
    current = np.zeros(N, dtype="bool")
    while True:
        score = evaluate(nk_table, K, current)
        if best_score < score:
            best_score = score
            best = np.copy(current)
        # Binary counter
        index = 0
        while index < N and current[index]:
            current[index] = False
            index += 1
        if index >= N:
            break
        current[index] = True
    return best


def dynamic_programming(nk_table, K):
    assert(nk_table.shape[1] == (2 << K))
    N = nk_table.shape[0]
    assert(K > 0)
    dependencies = 2 * K
    assert(N >= dependencies)
    patterns = 1 << dependencies
    K_plus_1_bits = (1 << (K + 1)) - 1
    powk = 1 << K
    powk_values = range(powk)
    # These are used to build and store the functions created
    # by removing a bit.
    previous_function = np.zeros(patterns)
    next_function = np.zeros(patterns)
    # Records how the removed bit should be set given its dependencies
    best_bit_choice = np.zeros((N, patterns), dtype="bool")

    # Creates a function to mimic all NK functions that wrap
    for i in range(K):
        # Used to strip off bits not used by this function
        f = N - K + i
        for pattern in range(patterns):
            # Leaves only the k+1 bits used by this function
            relevant_bits = (pattern >> i) & K_plus_1_bits
            previous_function[pattern] += nk_table[f, relevant_bits]

    # "n" is the bit position we are currently trying to remove.
    # You could also say at the end of this loop the problem size will be n.
    for n in range(N - 1, dependencies - 1, -1):
        print "Removing variable", n, "from the NK Table"
        for before_wrap in powk_values:
            for after_wrap in powk_values:
                # quality of setting bit "n" to zero and one.
                zero_quality, one_quality = 0, 0
                # The pattern the dependent bits take
                zero_pattern = before_wrap | (after_wrap << (K + 1))
                one_pattern = zero_pattern | powk
                # bit "n" only depends on two functions at this point:
                # NK function_{n-k} which uses the first K+1 bits and the
                # "previous_function"
                zero_quality += nk_table[n - K, zero_pattern & K_plus_1_bits]
                one_quality += nk_table[n - K, one_pattern & K_plus_1_bits]
                # previous_function doesn't use the lowest index bit
                zero_quality += previous_function[zero_pattern >> 1]
                one_quality += previous_function[one_pattern >> 1]
                # put the results into "next_function"
                next_function_entry = before_wrap | (after_wrap << K)
                if zero_quality > one_quality:
                    # Zero was better
                    next_function[next_function_entry] = zero_quality
                    best_bit_choice[n, next_function_entry] = False
                elif zero_quality == one_quality:
                    # tie
                    next_function[next_function_entry] = zero_quality
                    best_bit_choice[n, next_function_entry] = False
                    # print "A tie happened", n, zero_quality
                else:
                    # One was better
                    next_function[next_function_entry] = one_quality
                    best_bit_choice[n, next_function_entry] = True
        previous_function, next_function = next_function, previous_function

    # previous_function is now wide enough to score all remaining patterns
    # so we can just copy in the remaining NK functions
    for f in range(K):
        # Used to strip off bits not used by this function
        for before_wrap in powk_values:
            for after_wrap in powk_values:
                previous_function_pattern = before_wrap | (after_wrap << K)
                in_order = after_wrap | (before_wrap << K)
                in_use = (in_order >> f) & K_plus_1_bits
                previous_function[
                    previous_function_pattern] += nk_table[f, in_use]

    # Whichever previous_function entry has the highest quality
    # is the true best pattern
    best_pattern = 0
    for pattern in range(patterns):
        if previous_function[pattern] > previous_function[best_pattern]:
            # TODO Probably need a list of patterns to deal with ties
            best_pattern = pattern
    print "Expected:", previous_function[best_pattern] / N
    # Extract a global optimum
    solution = np.zeros(N, dtype="bool")
    k_bits = (1 << K) - 1
    after_wrap = best_pattern >> K
    before_wrap = best_pattern & k_bits
    # First "dependencies" bits comes from "best_pattern"
    dependency_pattern = after_wrap | (before_wrap << K)
    for i in range(dependencies):
        solution[i] = (dependency_pattern >> i) & 1

    # Use the previous "left" and "right" to recover the way to set bit i
    for i in range(dependencies, N):
        best_bit = best_bit_choice[i][before_wrap | (after_wrap << K)]
        solution[i] = best_bit
        # shift out old bit information, add in new bit
        before_wrap = (before_wrap >> 1) | (best_bit << (K - 1))

    return solution

if __name__ == "__main__":
    N = int(sys.argv[1])
    K = int(sys.argv[2])
    nk_table = np.random.rand(N, 2 << K)
    dynamic_answer = dynamic_programming(nk_table, K)
    print "Dynamic programming says quality is:", evaluate(nk_table, K, dynamic_answer)

    brute_answer = brute_force(nk_table, K)
    print "Brute force says quality is:       :", evaluate(nk_table, K, brute_answer)
    assert((dynamic_answer == brute_answer).all())
