def accumulate(func, sequence, starting_point):  # to be removed
    output = [starting_point]
    for element in sequence:
        output.append(func(element, output[-1]))
    return output[1:]
