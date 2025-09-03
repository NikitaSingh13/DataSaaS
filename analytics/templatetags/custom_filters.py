from django import template

register = template.Library()

@register.filter
def lookup(dictionary, key):
    """
    Template filter to lookup dictionary values by key
    Usage: {{ dictionary|lookup:key }}
    """
    if isinstance(dictionary, dict):
        return dictionary.get(key)
    return None

@register.filter
def replace(value, arg):
    """
    Template filter to replace characters in strings
    Usage: {{ value|replace:"_,:" }}
    """
    if isinstance(value, str) and arg:
        old, new = arg.split(',')
        return value.replace(old, new)
    return value

@register.filter
def dict_items(dictionary):
    """
    Template filter to get dictionary items for iteration
    Usage: {% for key, value in dictionary|dict_items %}
    """
    if isinstance(dictionary, dict):
        return dictionary.items()
    return []

@register.filter
def sub(value, arg):
    """
    Template filter to subtract two numbers
    Usage: {{ value|sub:10 }}
    """
    try:
        return int(value) - int(arg)
    except (ValueError, TypeError):
        return 0

@register.filter 
def mul(value, arg):
    """
    Template filter to multiply two numbers
    Usage: {{ value|mul:100 }}
    """
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return 0

@register.filter
def div(value, arg):
    """
    Template filter to divide two numbers
    Usage: {{ value|div:100 }}
    """
    try:
        return float(value) / float(arg) if float(arg) != 0 else 0
    except (ValueError, TypeError):
        return 0
