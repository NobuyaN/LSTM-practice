#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Nobuya Nishio
Student ID: 2600240478-7

Program Description: This program prints all the food contained in fav_foods 
list and prints the second element in the list.
"""


fav_foods = ["dan dan mian", "uni", "gyoza", "sushi"]
for food in fav_foods:
    print(food)
    
print("One of my favorite food is {}".format(fav_foods[1]))
