# Python progrom to find out the factorial of give numbers!!

num = int(input("Enter the number:"))
factorial = 1
if num <0:
  print("Factorial does not exit for negative number...")
elif num == 0:
  print("Factorial of zero is 1")
else:
  for i in range(1,num=1):
    factorial = factorial*i
  print(f'The factorial of {num} is {factorial}')
