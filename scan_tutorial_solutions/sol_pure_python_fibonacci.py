def fibonacci(n):
    output = [0, 1]
    for _ in range(2, n):
        output.append(output[-1] + output[-2])
    return output[:n]
