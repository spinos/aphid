from word import *
a = 'Casablanca.1942' 
print(a)
b = MY.Word()
a = b.reversed(a)
print(a)
a = b.str('alisa')
print(a)
n = ['alisa','bob', 'kate']
a = b.strvec(n)
print(a)
try:
    b.someerr()
except StandardError, e:
    print(e)

print('program ended')
