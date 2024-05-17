#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Nobuya Nishio
Student ID: 2600240478-7

Program Description: This program goes through two list in parallel and adds
 the respective element together, printing it, and then checks if the sum of the 
 two respective element is greater than 30. If true, it will break out of the
 loop.
"""


number_list_1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 10]
number_list_2 = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 20]

print("Length of first list:", len(number_list_1))
print("Length of second list:", len(number_list_2))

for number_1, number_2 in zip(number_list_1, number_list_2): #iterate through 2 list
    summa = number_1 + number_2 #add the respective element together
    print("summa: ", summa) #print summa 
    if summa > 30: #if summa is greater than 30, break from the loop
        break
    

        
        
    