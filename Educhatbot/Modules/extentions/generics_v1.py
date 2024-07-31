MATH_SYMBOLS = [
    ["+"],
    ["-", "−"],
    ["*", "×", "\\times"],
    ["/", "÷", "\\div"],
    ["+", "\\pm"],  # plus minus sign, just replace with plus now
    ["%", "\\%"],
    ["<=", "\\le"],
    [">=", "\\ge"],
    ["<", "\\lt"],
    [">", "\\gt"],
    ["!=", "\\ne"],
    ["π", "\\pi"],
    [" ° ", r"^{\circ}", r"^{\circ\ }", r"^{\circ }", r"^\circ", r"\degree"],
    ["^"],
    ["θ", "\\theta"],
    ["α", "\\alpha"],
    ["β", "\\beta"],
    ["γ", "\\gamma"],
    ["δ", "\\delta"],
    ["λ", "\\lambda"],

    # Trigo/Log functions using temporary variables: this is a hack and not proper eval, because the way answer texts
    # are currently written (as of Apr 2021)
    # ["ζ", "\\zeta", "sin", "cos", "tan", "cot", "sec", "cosec", "ln"],
    ["ι", "\\dots", "..."]
]

STANDARD_MATH_SYMBOLS = [symbols[0] for symbols in MATH_SYMBOLS]

SPECIAL_CHAR_EXCLUSIONS = [" ", "\\ ", ",", "\\", "\n", "{", "}", ":", " and ", ", and ", ";"]

SPECIAL_STANDARDIZE_CHARS = [
    [" ∠ ", r"\angle"],
    [" ° ", r"^{\circ}", r"^\circ"],
    ["%", "\\%"],
    [",", " and ", " & ", r" \(\&\) ", ", ", ",,", "<br>"],
    # [" & ", r" \(\&\) "],
    ["(", "\\("],
    [")", "\\)"],
    [" ", "\\ "],
]

STANDARDIZE_SYMBOLS = {
    **{symbols[0]: symbols for symbols in MATH_SYMBOLS},
    **{chars[0]: chars for chars in SPECIAL_STANDARDIZE_CHARS}
}

SPECIAL_STANDARDIZE_CHARS_EXACT_MATCH = [
    [" ∠ ", r"\angle"],
    [" ° ", "°", r"^{\circ}", r"^\circ", r"\degree"],
    ["%", "\\%"],
    [" & ", r" \(\&\) "],
    ["(", "\\("],
    [")", "\\)"],
    [" ", "\\ "],
    # units with slash which accidentally got misinterpreted
    ["m/s", "\\frac{m}{s}"],
    ["km/h", "\\frac{km}{h}"],
    ["cm/s", "\\frac{cm}{s}"],
    ["m/min", "\\frac{m}{min}"],
    ["g/cm3", "\\frac{g}{cm3}", "\\frac{g}{cm^3}", "\\frac{g}{cm^{3}}"],
]

STANDARDIZE_SYMBOLS_EXACT_MATCH = {
    **{symbols[0]: symbols for symbols in MATH_SYMBOLS},
    **{chars[0]: chars for chars in SPECIAL_STANDARDIZE_CHARS_EXACT_MATCH}
}

UNITS_LIST = [
    ["unit", "units"],
    # Length, Distances
    ["cm", "centimetre", "centimetres"],
    ["km", "kilometre", "kilometres"],
    ["mm", "millimetre", "millimetres"],
    ["m", "metre", "metres"],
    # Area, Areas
    ["centimetresquare", "cm2", "cm^2"],
    ["msquare", "m2", "m^2"],
    ["square"],
    # Volume
    ["ml", "mℓ", "millilitre", "millilitres"],
    ["l", "ℓ", "litre", "litres"],
    ["centimetrecube", "cm3", "cm^3"],
    ["decimetrecube", "dm3", "dm^3"],
    ["mcube", "m3", "m^3"],
    ["cube", "cubic"],
    # Mass
    ["kg", "kilogram", "kilograms"],
    ["mg", "milligram", "milligrams"],
    ["g", "gram", "grams"],
    # Speed (do first before time)
    ["mpersec", "m/s", "metre per sec", "metres per sec", "metre per second", "metres per second"],
    ["kmperhr", "km/h", "kilometre per hour", "kilometres per hour"],
    ["cmpersec", "cm/s", "centimetre per sec", "centimetres per sec"],
    ["mpermin", "m/min", "metre per minute", "metres per minute", "metre per min", "metres per min"],
    ["gpercm3", "g/cm^3", "g/cm3", "grams per centimetre", "gram per centimetre"],
    # Time
    ["s", "sec", "second", "seconds"],
    ["min", "mins", "minute", "minutes"],
    ["h", "hr", "hrs", "hour", "hours"],
    # Angle
    ["°", "degree", "degrees", "\\degree"],
    ["∠", "\\angle"],
    ["⊥"],
    # Currency
    ["$", "dollars", "dollar", "\\$"],
    ["₵", "cents", "cent", "\\₵"],
    # Time
    [" pm", " p.m.", " p.m"],
    [" am", " a.m.", " a.m"],
    # percent
    ["%", "\\%"],
    # numeric quantities
    ["ones"],
    ["tens"],
    ["hundreds"],
    ["thousands"],
]

UNITS = {unit[0]: unit for unit in UNITS_LIST}
STANDARDIZE_UNITS = {
    **{units[0]: units for units in UNITS_LIST},
}

# list of units with numbers. The order of this list is important, make sure no first element later in the list is
# substring of earlier elements
UNITS_WITH_NUMBERS_LIST = [
    # Area, Areas
    ["centimetresquare", "cm2", "cm^2", "cm^{2}", "cm^{2\ }", "cm^{2 }"],
    ["msquare", "m2", "m^2", "m^{2}", "m^{2 }", "m^{2\ }"],
    # Volume
    ["centimetrecube", "cm3", "cm^3", "cm^{3}", "cm^{3 }", "cm^{3\ }"],
    ["mcube", "m3", "m^3", "m^{3}", "m^{3 }", "m^{3\ }"],
]
UNITS_WITH_NUMBERS = {unit[0]: unit for unit in UNITS_WITH_NUMBERS_LIST}
STANDARDIZE_UNITS_WITH_NUMBERS = {
    **{units[0]: units for units in UNITS_WITH_NUMBERS_LIST},
}

UNIT_HALF_MARKS = {
    "ones": ["one"],
    "tens": ["ten"],
    "hundreds": ["hundred"],
    "thousands": ["thousand"],
}

MATHJAX_SYMBOLS = [
    ["\\frac"],
    ["\\sqrt"],
    ["\\text"],
    ["\\$"],
]

ALL_SYMBOLS = {
    **STANDARDIZE_SYMBOLS,
    **UNITS,
    **{mathjax[0]: [mathjax] for mathjax in MATHJAX_SYMBOLS},
}

MATHJAX_PYTHON_CONVERSION = {
    "\\sqrt": "cmath.sqrt",
    "sin": "cmath.sin",
    "cos": "cmath.cos",
    "π": "cmath.pi",
    "ln": "cmath.log",
    "^": "**",
    "\\(": "(",
    "\\)": ")",
    "{": "(",
    "}": ")",
    "\\pm": "+",  # use plus for plus-minus sign from math expression
}

NUMERIC_TOKENS = [
    "\\sqrt",
    "\\frac",
    ":",
    "/"
]

EXACTMATCH_PUNCTUATION_SYMBOLS = [
    ".",
    "?",
    "!",
    ",",
    ":",
    ";",
    "-",
    "–",
    "—",
    "(", ")",
    "[", "]",
    "{", "}",
    "<", ">",
    r'“', r'”', r'"',
    r"'",
    r"/",
    r"...",
    r"*",
    r"&",
    r"#",
    r"~",
    "\\",
    "@",
    "^",
    "|",
    "_"
]

EXACTMATCH_TOKENS = NUMERIC_TOKENS + EXACTMATCH_PUNCTUATION_SYMBOLS

# these trigger strings if appearing in the algebra expression, will trigger extended tests
ALGEBRA_EXTENDED_TEST_TRIGGERS = [
    ">=",
    "<=",
    ">",
    "<",
]
ALGEBRA_TEST_TRIES = 5
ALGEBRA_TEST_RANGE_LOW = -2.0
ALGEBRA_TEST_RANGE_HI = 2.0
ALGEBRA_EXTENDED_TEST_TRIES = 500
ALGEBRA_EXTENDED_TEST_RANGE_LOW = -20.0
ALGEBRA_EXTENDED_TEST_RANGE_HI = 20.0

# ALGEBRA_MULTIPART_SEPARATOR = [
#     " or ",
#     ",or ",
#     ", or "
# ]

TRIANGLE_TOKEN = ["TRIANGLE", "Triangle", "triangle"]
