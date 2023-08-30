# Exercise 1
text: str = 'Hello, world!'
percent: float = 3.14
is_connected: bool = False
people: list[str] = ['Mario', 'Luigi', 'James']


# Exercise 2
class Fruit:
    def __init__(self, name: str, grams: float) -> None:
        self.name = name
        self.grams = grams


def return_fruit_description(fruit: Fruit) -> str:
    return f'This {fruit.name} weighs {fruit.grams} grams.'


apple: Fruit = Fruit('Apple', 50)
description: str = return_fruit_description(apple)
print(description)


# Exercise 3
from typing import Optional

# Python 3.10+ = str | None
def get_user(user_id: int) -> Optional[str]:
    users: dict[int, str] = {0: 'Mario', 1: 'Luigi'}
    return users.get(user_id)


first_user: Optional[str] = get_user(0)
print(first_user )
