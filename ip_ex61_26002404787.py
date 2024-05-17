#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Nobuya Nishio
Student ID: 2600240478-7

Program Description: This program would first merge the two list together into
one by the addition sign. It would then go through the names in scientist_list_3
to be appended to scientist_list. It would then print the length of the list with 
the len() function. It would then reverses the list by using the reverse() method.
It is then initialized with a new dictionary called scientist_dict, which would 
use the enumerate function to assign the index as the value and the name as the keys.
It would then use the update method to merge the two dictionaries together. 
Index list would then store the values of the dictionary, which would be sliced
only for the first 6 element. It would then be filtered to remove odd numbers 
by using modulus and the remove method. The variable 'one' would store the popped 
elemented of numbers in index 0. The maximum number in 'numbers' would then be 
printed using the max function. 
"""


scientist_list_1 = ['Newton','Einstein', 'Curie', 'Darwin']
scientist_list_2 = ['Tesla', 'Galilei', 'Lovelace']

scientist_list = scientist_list_1 + scientist_list_2

scientist_list_3 = ["Faraday", "Hawking"]

for name in scientist_list_3:
    scientist_list.append(name)

print("Length of the scientist list:", len(scientist_list))

scientist_list.reverse()
print("Reversed scientist list:", scientist_list)

scientist_dict = {}

for i, name in enumerate(scientist_list):
    scientist_dict[name] = i

scientist_dict_2 = {"Faraday":22, "Boyle":9}
scientist_dict.update(scientist_dict_2)

print("The scientist dictionary:", scientist_dict)

index_list = list(scientist_dict.values())

numbers = index_list[:6]
print("Numbers list:", numbers)

for i in numbers:
    if i % 2 != 0:
        numbers.remove(i)
        

one = numbers.pop(0)

print("The numbers list filtered out of odd numbers:", numbers)
print("The maximum value in the numbers list:", max(numbers))
