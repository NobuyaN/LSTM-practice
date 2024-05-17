#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Nobuya Nishio
Student ID: 2600240478-7

Program Description: This program prints the grade based on the points that 
the student got. In this case, the student gas a grade as 87.
"""

points = 87 

if points >= 89:
    print("A+")
elif points >= 80:
    print("A")
elif points >= 69:
    print("B")
elif points >= 59:
    print("C")
elif points < 55: # print F when it is less than 55
    print("F")  
    