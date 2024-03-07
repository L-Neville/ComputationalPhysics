def Fibonacci(n, memo={}):
    # memo is a memoization that stores previous values to prevent stack overflow
    if n <= 1:
        return n
    memo[n] = Fibonacci(n-1) + Fibonacci(n-2)
    return memo[n]

n = [7, 15, 31]
fn = [Fibonacci(i) for i in n]
for i, e in enumerate(n):
    print(f"the {e}th Fibonacci number is {fn[i]}")