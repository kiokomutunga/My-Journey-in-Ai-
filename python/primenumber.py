# Function to check if a number is prime
def is_prime(num):
    if num <= 1:
        return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    return True

# input for a limit
limit = int(input("Enter the last number you need the prime numbers to count to: "))

print(f"The prime numbers up to {limit} are:")
for n in range(2, limit + 1):
    if is_prime(n):
        print(n, end=" ")  # <-- make sure this is indented under `if`
