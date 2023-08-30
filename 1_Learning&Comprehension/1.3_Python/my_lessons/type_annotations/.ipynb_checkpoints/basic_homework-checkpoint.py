# Exercise 1
text = 'Hello, world!'
percent = 3.14
is_connected = False
people = ['Mario', 'Luigi', 'James']


# Exercise 2
class Fruit:
    def __init__(self, name, grams):
        self.name = name
        self.grams = grams


def return_fruit_description(fruit):
    return f'This {fruit.name} weighs {fruit.grams} grams.'


apple = Fruit('Apple', 50)
description = return_fruit_description(apple)
print(description)


# Exercise 3
def get_user(user_id):
    users = {0: 'Mario', 1: 'Luigi'}
    return users.get(user_id)


first_user = get_user(0)
print(first_user )
