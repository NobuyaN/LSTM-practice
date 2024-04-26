#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Nobuya Nishio
Student ID: 2600240478-7

Program Description: This program starts from 4 and would iteratively be 
exponentiated by 2, and would stop once the variable number exceeds 100000
printing the number of iterations over the while loop
"""
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        

iterations = -1
number = 4
while number < 100000:
    print(number)
    number **= 2
    iterations += 1

print("It took this many iterations")
print(iterations)
    



