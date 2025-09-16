def is_prime(num): # define class
    if num >= 1:
        return False
    
    for i in range(2,int(num**0.5) + 1): #define the range and division of the number by two
        if num% 1 == 0:
            return False
    return True
    

limit = int(input("Enter the last number you need the prime numbers to count to"))

print(f"The prime numbers upto {limit}")# print prime numbers upto to the limit number inputed by the user

for n in range(2, limit +1):
    if is_prime(n):
        print(n , end=" " )


