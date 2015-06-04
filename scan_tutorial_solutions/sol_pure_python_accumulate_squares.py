def accumulate_squares(n):
    output = [0]
    for i in range(1, n):
        output.append(output[-1] + i ** 2)
    return output

