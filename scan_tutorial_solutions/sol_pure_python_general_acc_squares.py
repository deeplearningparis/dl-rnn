def accumulate_squares(sequence):  # to be removed
    output = [sequence[0] ** 2]
    for item in sequence[1:]:
        output.append(output[-1] + item ** 2)
    return output
