# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 15:56:38 2017

@author: Obaid Gill
"""
##################
###INTRODUCTION###
##################

#Say "Hello, World!" With Python
if __name__ == '__main__':
    print "Hello, World!"
    
#Python If-Else
if __name__ == '__main__':
    n = int(raw_input())
    if n%2 != 0:
        print "Weird"
    elif n%2 == 0 and n in range(2,5+1):
        print "Not Weird"
    elif n%2 == 0 and n in range(6,20+1):
        print "Weird"
    elif n%2 == 0 and n > 20:
        print "Not Weird"
    else:
        pass

#Arithmetic Operators
if __name__ == '__main__':
    a = int(raw_input())
    b = int(raw_input())
    print(a+b)
    print(a-b)
    print(a*b)    

#Python: Division
from __future__ import division
if __name__ == '__main__':
    a = int(raw_input())
    b = int(raw_input())
    print(a//b)
    print(a/float(b))

#Loops    
if __name__ == '__main__':
    n = int(raw_input())
    for i in range(0,n):
        print(i*i)

#Write a function
def is_leap(year):
    leap = False
    leap = (year%4 == 0 and year%100 != 0) or (year%400 == 0) 
    return leap  
      
#Print Function
from __future__ import print_function
if __name__ == '__main__':
    n = int(raw_input())
    print(int(''.join([str(i) for i in range(1,n+1)])))    


######################
###BASIC DATA TYPES###
######################

#Lists
if __name__ == '__main__':
    N = int(raw_input())
    LO = []
    LI = []
    for i in range(N):
        LI.append(str(raw_input()))
    
    for i in range(len(LI)):
        if LI[i][0:3] == 'ins':
            LO.insert(int(LI[i].split(' ')[-2]),int(LI[i].split(' ')[-1]))
        elif LI[i][0:3] == 'pri':
            print (LO)
        elif LI[i][0:3] == 'rem':
            LO.remove(int(LI[i].split(' ')[-1]))        
        elif LI[i][0:3] == 'app':
            LO.append(int(LI[i].split(' ')[-1]))
        elif LI[i][0:3] == 'sor':
            LO.sort()
        elif LI[i][0:3] == 'pop':
            LO.pop()
        elif LI[i][0:3] == 'rev':
            LO.reverse()
        else:
            pass
#Tuple
if __name__ == '__main__':
    n = int(raw_input())
    integer_list = map(int, raw_input().split())
    print (hash(tuple(integer_list)))        

#List Comprehensions
if __name__ == '__main__':
    x = int(raw_input())+1
    y = int(raw_input())+1
    z = int(raw_input())+1
    n = int(raw_input())
    print ([[i,j,k] for i in range(x) for j in range(y) for k in range(z) if (i+j+k)!=n])

#Find the Second Largest Number           
if __name__ == '__main__':
    n = int(raw_input())
    arr = map(int, raw_input().split())
    print (sorted([i for i in arr if i < max(arr)])[-1])

#Nested Lists    
List = []
for _ in range(int(raw_input())):
    name = raw_input()
    score = float(raw_input())
    List.append([name,score])

for i in sorted([j[0] for j in List if j[1] == sorted([i[1] for i in List])[1]]):
    print (i)     

#Finding the percentage
n = int(raw_input())
student_marks = {}
for _ in range(n):
    line = raw_input().split()
    name, scores = line[0], line[1:]
    scores = map(float, scores)
    student_marks[name] = scores
query_name = raw_input()
 
print (format(sum(student_marks[query_name])/len(student_marks[query_name]),'.2f'))

       
#############
###STRINGS###
#############

#sWAP cASE
def swap_case(s):
    sout = ''
    for i in range(len(s)):
        if s[i].isupper():
            sout += s[i].lower()
        else:
            sout += s[i].upper()
    return (sout) 
    
#String Split and Join
def split_and_join(line):
    line = line.split(" ")
    return ("-".join(line))

#What's Your Name?
def print_full_name(a, b):
    print ("Hello " + a + " " + b + "! You just delved into python.")
    
#Mutations
def mutate_string(string, position, character):
    lstring = list(string) 
    lstring[position-1] = character
    return ("".join(lstring))

#Find a string
def count_substring(string, sub_string):
    iList = []
    for i in range(0, len(string)):
        if string[i:len(sub_string)+i] == sub_string:
            iList.append(string[i:len(sub_string)+i])
        else:
            pass
    return(len(iList))

#String Validators
if __name__ == '__main__':
    s = raw_input()
    alnum = ''
    alpha = ''
    digit = ''
    lower = ''
    upper = ''
    for i in range(0,len(s)):

        if s[i].isalnum():
            alnum += 'True'
        else:
            alnum += 'False'

        if s[i].isalpha():
            alpha += 'True'
        else:
            alpha += 'False'

        if s[i].isdigit():
            digit += 'True'
        else:
            digit += 'False'

        if s[i].islower():
            lower += 'True'
        else:
            lower += 'False'

        if s[i].isupper():
            upper += 'True'
        else:
            upper += 'False'
    
print('True' in alnum)
print('True' in alpha)
print('True' in digit)
print('True' in lower)
print('True' in upper)

#Text Alignment
#Replace all ______ with rjust, ljust or center. 

thickness = int(raw_input()) #This must be an odd number
c = 'H'


for i in range(thickness):#Top Cone
    print (c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1)        
        
for i in range(thickness+1):#Top Pillars
    print (c*thickness).center(thickness*2)+(c*thickness).center(thickness*6)

for i in range((thickness+1)/2):#Middle Belt
    print (c*thickness*5).center(thickness*6)    

for i in range(thickness+1):#Bottom Pillars
    print (c*thickness).center(thickness*2)+(c*thickness).center(thickness*6)  

for i in range(thickness):#Bottom Cone
    print ((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6)   

#Text Wrap
def wrap(string, max_width):
    return (textwrap.fill(string,max_width))

#Designer Door Mat
N, M = map(int,raw_input().split()) # More than 6 lines of code will result in 0 score. Blank lines are not counted.
for i in xrange(1,N,2): 
    print ('.|.'*i).center(N*3,'-')
print ('WELCOME').center(N*3,'-')
for i in xrange(N-2,-1,-2): 
    print ('.|.'*i).center(N*3,'-')

#String Formatting
def print_formatted(number):
    n = int(number)
    width = len("{0:b}".format(n))
    for i in xrange(1,n+1):
        print ("{0:{width}d} {0:{width}o} {0:{width}X} {0:{width}b}".format(i, width=width))

#Alphabet Rangoli      
def print_rangoli(size):
    a = 'abcdefghijklmnopqrstuvwxyz'
    if size > 1:
        for i in range(size-1,-1,-1):
            print ("-".join(a[i:size][::-1]) + "-" + ("-".join(a[i+1:size]))).center(size*3 + size-3,'-')   
            
        for i in range(1,size,1):
            print ("-".join(a[i:size][::-1]) + "-" + ("-".join(a[i+1:size]))).center(size*3 + size-3,'-') 
    else:
        print (a[size-1])

#Capitalize!
def capitalize(string):
    istring = list(string.split(' '))
    ostring = ''
    count = 0
    for i in range(len(istring)):
        if istring[i].isalnum():
            ostring +=''.join(list(istring[i])[0].upper()) + ''.join(list(istring[i])[1:]) + ' ' 
        else:
            ostring += ' '
    return(ostring.rstrip())
        
#Merge the Tools
def merge_the_tools(string, k):
    istring = list(string)
    seg = range(0,k*k+k,k) 
    
    for i in range(len(seg)-1):
        tstring = []
        if len(seg)-1 == 1:
            tstring = istring
        else:
            tstring = istring[seg[i]:seg[i+1]]
        
        ostring = []
        for j in range(len(tstring)):
            if tstring[j] in ostring:
                pass
            else:
                ostring.append(tstring[j])
                
        print(''.join(ostring))
    
  
##########
###SETS###
##########
#Introduction to Sets
def average(array):
    return(float(sum(set(array)))/float(set(len(array)))

#Symmetric Difference
M = int(raw_input())
Marr = set(map(int,raw_input().split()))
N = int(raw_input())
Narr = set(map(int,raw_input().split()))

for i in sorted(Marr.difference(Narr).union(Narr.difference(Marr))):
    print (i)

#No Idea!
NM = raw_input().split()
Arr = raw_input().split()
MarrA = set(raw_input().split())
MarrB = set(raw_input().split())
count = 0
for i in Arr:
    if i in MarrA and i in MarrB:
        count = count + 0
    elif i in MarrA and i not in MarrB:
        count = count + 1
    elif i not in MarrA and i in MarrB:
        count = count - 1
    else:
        pass

print (count)

#Set .add()
N = int(raw_input())
s = set()
for i in range(N):
    s.add(raw_input())
print len(s)

#Set .discard(), .remove() & .pop()
n = input()
SO = set(map(int,raw_input().split(' ')))
N = int(raw_input())
SI = []
for i in range(N):
    SI.append(str(raw_input()))
  
for i in range(len(SI)):
    if SI[i][0:3] == 'pop':
        SO.pop()     
    elif SI[i][0:3] == 'rem':
        SO.remove(int(SI[i].split(' ')[-1]))     
    elif SI[i][0:3] == 'dis':
        SO.discard(int(SI[i].split(' ')[-1]))        
    else:
       pass
for i in SO:
    print i
    
#Set .union() Operation
n = int(raw_input())
nE = set(raw_input().split())

b = int(raw_input())
nF = set(raw_input().split())

print len(nE.union(nF))

#Set .intersection() Operation
n = int(raw_input())
nE = set(raw_input().split())

b = int(raw_input())
nF = set(raw_input().split())

print len(nE.intersection(nF))

#Set .difference() Operation
n = int(raw_input())
nE = set(raw_input().split())

b = int(raw_input())
nF = set(raw_input().split())

print len(nE.difference(nF))

#Set .symmetric_difference() Operation
n = int(raw_input())
nE = set(raw_input().split())

b = int(raw_input())
nF = set(raw_input().split())

print len(nE.symmetric_difference(nF))

#The Captain's Room
K = int(raw_input())
Room = map(int,raw_input().split())
for i in set(Room):
    if Room.count(i) == 1:
        break 
print (i)

###########
###MATHS###
###########

#Triangle Quest 2
for x in range(1,int(input())+1):
    print(((10**x - 1)//9)**2)
    
for j in range(1,int(raw_input())+1):#OR
    print int(''.join([str(i) for i in range(1,j+1)]) + ''.join([str(i) for i in range(1,j)[::-1]]))    

#Mod Divmod
a = int(raw_input())
b = int(raw_input())

print a/b
print a%b
print (divmod(a,b)) 

#Power - Mod Power
a = int(raw_input())
b = int(raw_input())
m = int(raw_input())
print pow(a,b)
print pow(a,b,m)

#Integers Come In All Sizes
a = int(raw_input())
b = int(raw_input())
c = int(raw_input())
d = int(raw_input())

print (a**b)+(c**d)
    
#Triangle Quest
for i in range(1,input()):
    print (int(str(i)*i))
    
for i in range(1,input()): #OR
    print(((10**i - 1)//9)*i)

###############
###ITERTOOLS###
###############
#itertools.product()
from itertools import product
A = map(int,raw_input().split(' '))
B = map(int,raw_input().split(' '))

for i in (list(product(A,B))):
    print (i),        

#itertools.permutations()
from itertools import permutations
S = raw_input().split(' ')
SO = []
if len(S)>1:
    for i in (list(permutations(S[0],int(S[1])))):
        SO.append(''.join(i))
    for i in sorted(SO):
        print i
else:
    for i in (list(permutations(S[0]))):
        SO.append(''.join(i))
    for i in sorted(SO):
        print i

#itertools.combinations()
from itertools import combinations
S = raw_input().split(' ')

SI = ''.join(sorted(list(S[0])))
N = range(1,int(S[1])+1)
SO = []

for i in N:
    for j in (list(combinations(SI,i))):
        SO.append(''.join(j))

for i in SO:
    print (i)    

#itertools.combinations_with_replacement()
from itertools import combinations_with_replacement
S = raw_input().split(' ')

SI = ''.join(sorted(list(S[0])))
N = range(1,int(S[1])+1)
SO = []

for i in N:
    for j in (list(combinations_with_replacement(SI,i))):
        SO.append(''.join(j))

for i in SO:
    if len(i)==N[-1]:
            print (i)   

#Compress the String!
from itertools import groupby
S = raw_input()

SI = [list(v) for k, v in groupby(list(S))] #no value exists only keys available
 
for i in SI:
    print "("+str(len(i))+", "+str(i[0])+")",

#Maximize It!
from itertools import product 

from itertools import product
k,m  = map(int,raw_input().strip().split(" "))

l = []
for _ in range(k):
    l.append([i**2 for i in map(int,raw_input().strip().split(" "))[1:]])

print (max([sum(i)%m for i in list(product(*l))]))

#################
###Collections###
#################
#collecions.Counter()
from collections import Counter
from itertools import groupby
import operator
X = int(raw_input())
N = raw_input().split(' ')
Xi = int(raw_input())
SRan = []
for a in range(Xi):
    SRan.append(raw_input().split(' '))
    
    
CN = Counter(N)
SReq =  [(i,j) for i,j in SRan if i in CN.keys()]

def accumulate(l):#stack overflow help
    it = groupby(l, operator.itemgetter(0))
    for key, subiter in it:
       yield key, [item[1] for item in subiter] 

SAva = []
for k,v in CN.items():
    if k in dict(list(accumulate(SReq))).keys():
        SAva.append(dict(list(accumulate(SReq)))[k][:v])

print sum([int(item) for sublist in SAva for item in sublist])           
           
#DefaultDict Tutorial
n,m = map(int,raw_input().split(' '))

nL = []
for i in range(n):
    nL.append(raw_input())

mL = []
for j in range(m):
    mL.append(raw_input())
    
for k in mL:
    print ' '.join(map(str,([l+1 for l,m in list(enumerate(nL)) if m == k])))  

#Collections.namedtuple()
N = int(raw_input())
C_ENTRY = list(raw_input().split(' '))

LIST = [] 
for i in range(N):
    LIST.append(raw_input().split(' '))

MARKS = []
for i in LIST:
    MARKS.append(i[C_ENTRY.index('MARKS')])

SUM_MARKS = sum([int(i) for i in MARKS])    
print (SUM_MARKS/N)

#Collections.OrderedDict()
from collections import OrderedDict 
N = int(raw_input())  
SL = []
for i in range(N):
    SL.append(raw_input().split(' '))
    
ordered_dictionary = OrderedDict() 
for i in range(len(SL)):
    if ' '.join(SL[i][:-1]) not in ordered_dictionary.keys():
        ordered_dictionary[' '.join(SL[i][:-1])] = int(SL[i][-1]) 
    else:
        ordered_dictionary[' '.join(SL[i][:-1])] += int(SL[i][-1])

for i in ordered_dictionary:
    print i + ' ' + str(ordered_dictionary[i])
           
#Word Order
from collections import OrderedDict 
N = int(raw_input())  
SL = []
for i in range(N):
    SL.append((''.join(raw_input().split(' ')),1))

ordered_dictionary = OrderedDict() 
for i in range(len(SL)):
    if ' '.join(SL[i][:-1]) not in ordered_dictionary.keys():
        ordered_dictionary[' '.join(SL[i][:-1])] = SL[i][-1] 
    else:
        ordered_dictionary[' '.join(SL[i][:-1])] += SL[i][-1]
        
print (len(ordered_dictionary.keys())) 

for i in ordered_dictionary:
    print str(ordered_dictionary[i]),  
           
#Collections.deque()
from collections import deque
N = int(raw_input())
DQ = deque()
LI = []
for i in range(N):
    LI.append(str(raw_input()))

for i in range(len(LI)):
    if LI[i][:-2] == 'append':
        DQ.append(int(LI[i].split(' ')[-1]))
    elif LI[i][:-2] == 'appendleft':
        DQ.appendleft(int(LI[i].split(' ')[-1]))
    elif LI[i][:-2] == 'p':
        DQ.pop()  
    elif LI[i][:-2] == 'pople':
        DQ.popleft()    
    else:
        pass
for i in DQ:
    print i,

#Most Common
from collections import OrderedDict 
S = list(raw_input()) 

SL = []
for i in S:
    SL.append((i,1))

ordered_dictionary = OrderedDict() 
for i in range(len(SL)):
    if SL[i][0] not in ordered_dictionary.keys():
        ordered_dictionary[SL[i][0]] = SL[i][-1] 
    else:
        ordered_dictionary[SL[i][0]] += SL[i][-1] 
    
for i in sorted(ordered_dictionary.items(), key=lambda x: (-x[1], x[0]))[0:3]:
    print str(i[0]) +' '+ str(i[1]) 
           
           
#################
##Date and Time##
#################
#Calendar Module           
from datetime import date
import calendar
d = raw_input().split(' ')
d = [i.lstrip('0') for i in d]
d.reverse()  
d = map(int,d)   

my_date = date(d[0],d[2],d[1]) #y m d
dOut = calendar.day_name[my_date.weekday()]
print dOut.upper()  
           
#Time Delta
N = int(raw_input())
t = []
for i in range(N*2):
    t.append(raw_input())   

from dateutil.parser import parse
for i,j in zip([a+1 for a in range(2*2) if (a+1)%2!=0],[b+1 for b in range(2*2) if (b+1)%2==0]):
    mytime = parse(t[i-1], fuzzy=True) -parse(t[j-1], fuzzy=True)
    print int(abs(mytime.total_seconds()))

#######################
#Errors and Exceptions#
#######################
#Exceptions
N = int(raw_input())
for i in range(N):
    a,b = raw_input().split(' ')
    try:
        print int(a)/int(b)        
    except ValueError as e:
        print "Error Code:",e  
    except ZeroDivisionError as e:
        print "Error Code:",e 
           
#Incorrect Regex
import re
N = int(raw_input())
for _ in range(N):
    a = raw_input()
    try:
        re.compile(a)
        print "True"
    except:
        print "False"           
           
#############
##Built-Ins##
#############
#Zipped!           
from itertools import product
N,X = map(int,raw_input().split(' '))

L = []
for i in range(X):
    L.append(map(float,raw_input().split(' ')))
 
iterate = sorted(list(product(range(X),range(N))), key = lambda x : x[1])  

L2= []
for i,j in iterate:
    L2.append(L[i][j])

for i in [L2[i:i+X] for i in xrange(0, len(L2), X)]:
    print sum(i)/len(i)
           
#Input()
x,k = map(int,raw_input().split(' '))
print eval(raw_input()) == k

#Python Evaluation
from __future__ import print_function
eval(raw_input()) 

#Sort Data
N,M = map(int,raw_input().split(' '))

L = []
for i in range(N):
    L.append(map(int,raw_input().split(' ')))

K = int(raw_input())

LS = sorted(L, key = lambda x: x[K])    

for i in range(len(LS)):
    print (' '.join(map(str,(LS[i]))))     
           
#Any or All
N = int(raw_input())
M = map(int,raw_input().split(' '))

if all([i>0 for i in M]):
    if any([i == j for i,j in zip(M,[int(''.join(list(str(i))[::-1])) for i in M])]):
        print ('True')
else:
    print ('False') 

########################           
###Python Functionals###           
########################           
#Map and Lambda Function
cube = lambda x: x**3

def fibonacci(n):
    count = 0
    L = [0,1]
    while (count < n):
        L.append(L[len(L)-2] + L[len(L)-1])
        count += 1
    return (L[0:n])

#Validating Email Addresses With a Filter 
def fun(s):
    import re
    return(len(re.findall(r'^[a-z0-9_\-]+@\b\w+[.]\w+$',s))>0) 

#Reduce Function
def product(fracs):
    t = reduce(lambda x, y : x*y, fracs) 
    return t.numerator, t.denominator   
           
###########
###Numpy###
###########
#Arrays
def arrays(arr):
    import numpy
    return(numpy.array(list(arr), float)[::-1])
           
#Shape and Reshape
import numpy
print (numpy.reshape(numpy.array(map(int,(raw_input().split(' ')))),(3,3)))
           
#Transpose and Flatten
import numpy
n,m = map(int,raw_input().split(' '))

L = []
for _ in range(n):
    L.append(map(int, raw_input().split(' ')))
my_array = numpy.array(L)    
print(numpy.transpose(my_array))
print(my_array.flatten())
       
#Concatenate
import numpy
n,m,p = map(int, raw_input().split(' '))
L = []
for _ in range(n+m):
    L.append(map(int, raw_input().split(' ')))
my_array = numpy.array(L) 
print (my_array)
           
#Zeros and Ones
import numpy
n = map(int,raw_input().split(' '))
nI= []
for i in range(len(n)):
    nI.append (n[i])
print (numpy.zeros(tuple(nI), dtype = numpy.int))   
print (numpy.ones(tuple(nI), dtype = numpy.int))  
           
#Eye and Identity
import numpy
n,m = map(int,raw_input().split(' '))
print numpy.eye(n,m, k = 0)
           
#Array Mathematics
import numpy as np
n, m = map(int, input().split())
a, b = (np.array([input().split() for _ in range(n)], dtype=int) for _ in range(2))
print(a+b, a-b, a*b, a//b, a%b, a**b, sep='\n')
           
#Floor, Ceil and Rint
import numpy
my_array = map(float,raw_input().split(' ')) 
print (numpy.floor(my_array))
print (numpy.ceil(my_array))
print (numpy.rint(my_array)) 
           
#Sum and Prod
import numpy
n,m = map(int,raw_input().split(' '))
L = []
for i in range(n):
    L.append(map(int,raw_input().split(' ')))
print (numpy.prod(numpy.sum(L, axis = 0)))
           
#Min and Max 
import numpy
n,m = map(int,raw_input().split(' '))
L = []
for i in range(n):
    L.append(map(int,raw_input().split(' ')))
print (numpy.max(numpy.min(L, axis = 1)))
           
#Mean, Var, and Std
import numpy
n,m = map(int,raw_input().split(' '))
L = []
for i in range(n):
    L.append(map(int,raw_input().split(' ')))

print (numpy.mean(L, axis = 1))
print (numpy.var(L,axis = 0))    
print (numpy.std(L))
           
#Dot and Cross 
import numpy
n = int(raw_input())
A = []
for i in range(n):
    A.append(map(int,raw_input().split(' ')))
B = []
for i in range(n):
    B.append(map(int,raw_input().split(' ')))    

print (numpy.dot(A,B))           
           
#Inner and Outer
import numpy
A = map(int,raw_input().split(' '))        
B = map(int,raw_input().split(' '))    

print (numpy.inner(A,B))    
print (numpy.outer(A,B))             

#Polynomials  
import numpy
print (numpy.polyval(map(float,raw_input().split(' ')) , int(raw_input())))           
           
#Linear Algebra
import numpy
n = int(raw_input())
A = []
for i in range(n):
    A.append(map(float,raw_input().split(' ')))
print numpy.linalg.det(A)            
           
           
           
  

