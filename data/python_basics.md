# Python Programming Basics

Python is a high-level, interpreted programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991.

## Why Python?

- **Easy to learn**: Simple syntax similar to English
- **Versatile**: Used for web development, data science, AI, automation
- **Large community**: Extensive libraries and frameworks
- **Cross-platform**: Runs on Windows, Mac, Linux

## Basic Syntax

### Variables and Data Types

```python
# Variables
name = "John"
age = 25
height = 5.9
is_student = True

# Data types
# String: str
# Integer: int
# Float: float
# Boolean: bool
# List: list
# Dictionary: dict
```

### Control Flow

```python
# If-else statement
if age >= 18:
    print("Adult")
else:
    print("Minor")

# For loop
for i in range(5):
    print(i)

# While loop
count = 0
while count < 5:
    print(count)
    count += 1
```

### Functions

```python
def greet(name):
    return f"Hello, {name}!"

message = greet("World")
print(message)  # Output: Hello, World!
```

## Popular Libraries

1. **NumPy** - Numerical computing
2. **Pandas** - Data manipulation and analysis
3. **Matplotlib** - Data visualization
4. **Django/Flask** - Web development
5. **TensorFlow/PyTorch** - Machine learning
6. **Requests** - HTTP library

## Best Practices

- Use meaningful variable names
- Follow PEP 8 style guide
- Write docstrings for functions
- Use virtual environments for projects
- Handle exceptions properly
- Write unit tests
