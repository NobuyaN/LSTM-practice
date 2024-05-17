#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Nobuya Nishio
Student ID: 2600240478-7

Program Description: This program goes from 1 to 400 and adds up all the number
that is even (remained of 0 when doing modulus of 2), while number is incremented
by 1 each loop.
"""

number = 1
total = 0

while number <= 400: # iterate the number until it is above 400
    if number % 2 == 0: # check if the number is even
         total += number # add the even number to total and accumulate it 
    number += 1 #increment number by 1
         
print(total)
    
