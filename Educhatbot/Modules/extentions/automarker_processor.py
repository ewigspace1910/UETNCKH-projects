from Modules.extentions import generics_v1 as Generics
import re as regex

decimal_tolerance = 0.0001


def remove_nested_braces(text):
    # Define the pattern for nested braces
    pattern1 = r"(_{)+"
    pattern2 = r"\}\}+"

    # Replace all occurrences of the pattern with an empty string
    text = regex.sub(pattern1, "", text)
    text = regex.sub(pattern2, "", text)
    return text
def clean_question_text(question):
  question = remove_nested_braces(question)
  # question =_process_answer_string_algebra(question)
  question = _insert_multiplication_sign_before_sqrt(question)
  # question = _process_space_around_numbers(question)
  # question = _process_space_around_numbers(question, True)
  question = preprocess_answer_string(question).replace("\\", " \\")
  return question


def is_number(token: str) -> bool:
    if token.isnumeric():
        return True
    else:
        try:
            float(token)
        except ValueError:
            return False

        return True


def is_numeric_token(token: str) -> bool:
    return is_number(token) or token in Generics.NUMERIC_TOKENS


def is_literal(token: str) -> bool:
    return token.isalpha()


def get_numerals(tokens: list) -> list:
    return [float(token) if is_number(token) else token for token in tokens if is_numeric_token(token)]


def get_literals(tokens: list, exclusions: dict = {}) -> list:
    return [token for token in tokens
            if not any(token in exclusion for exclusion in exclusions.values()) and is_literal(token)]


def get_units(tokens: list, units: dict) -> list:
    return [token for token in tokens if any(token in unit for unit in units.values())]


def get_units_algebra(tokens: list, units: dict) -> list:
    # this function only gets units from back of string, excep dollar sign ($) which is front of string.
    # The rest middle of string is not a unit
    output = []
    for unit in units.values():
        if unit[0] == '$' and tokens[0] in unit:
            output.append(tokens[0])
        elif unit[0] != '$' and tokens[-1] in unit:
            output.append(tokens[-1])
        else:
            pass

    return output


def is_numbers_same(a: list, b: list) -> bool:
    if len(a) != len(b):
        return False
    if not a or not b:
        return False

    try:
        is_element_same = [abs(float(valA) - float(valB)) < decimal_tolerance if is_number(str(valA)) and is_number(str(valB))
                           else valA == valB for (valA, valB) in zip(a, b)]
    except ValueError:
        return False

    return all(is_element_same)


def is_words_same(a: list, b: list) -> bool:
    if len(a) != len(b):
        return False
    else:
        return all([valA == valB for (valA, valB) in zip(a, b)])


def is_exact_match(a: list, b: list) -> bool:
    return a == b


def _remove_ending_escape_char(answer_text: str) -> str:
    if answer_text.endswith('\\'):
        return answer_text[:-1]
    else:
        return answer_text


def _remove_enclosing_brackets_and_whitespace(answer_text: str) -> str:
    old_answer_text = answer_text
    while(True):
        if answer_text.startswith(r"\(") and answer_text.endswith(r"\)"):
            new_answer_text = answer_text[2:-2].strip()
        else:
            new_answer_text = answer_text.strip()

        # repeat until no further changes
        if new_answer_text == old_answer_text:
            return new_answer_text
        else:
            old_answer_text = new_answer_text


def _remove_text_tags_from_answer_text(answer_text: str) -> str:
    text_tag_open = r"\text{"
    return _remove_tag_from_answer_text(answer_text, text_tag_open)


def _remove_tag_from_answer_text(answer_text: str, tag: str) -> str:
    while tag in answer_text:
        index = answer_text.find(tag)
        text_begin = answer_text[:index]
        text_middle = ""
        text_back = ""
        text_temp = answer_text[index + 6:]

        level = 0
        for char in text_temp:
            if level < 0:
                text_back = text_back + char
            else:
                if char == "{":
                    level += 1
                elif char == "}":
                    level -= 1

                if level >= 0:
                    text_middle = text_middle + char

        answer_text = text_begin + text_middle + text_back

    return answer_text


def _remove_left_right_tags_from_answer_string(answer_text: str) -> str:
    return answer_text.replace("\\left", "").replace("\\right", "")


def _remove_br_tags_from_answer_string(answer_text: str) -> str:
    if answer_text.lower().startswith("<br>"):
        return answer_text[len("<br>"):].strip()
    else:
        return answer_text


def process_and_split_answer_string(answer_text: str,
                                    insert_space: bool = False,
                                    standardize_units: bool = False) -> list:

    # preprocess units with numbers like cm3, m3 etc so that they dont get added space (cm2 = cm 2) after this
    if insert_space and standardize_units:
        answer_text = _standardize_symbols_string(answer_text, Generics.STANDARDIZE_UNITS_WITH_NUMBERS)

    answer_text = _process_space_around_numbers(answer_text,
                                                insert_space_between_number_letter=insert_space,
                                                remove_space_between_numbers_or_decimals_or_slash=True)

    answer_text = _standardize_symbols_string(answer_text, Generics.STANDARDIZE_SYMBOLS_EXACT_MATCH, insert_space=insert_space)

    # replace braces with division (for frac) or spaces
    answer_text = answer_text.replace('}{', ' / ')
    answer_text = answer_text.replace('{', ' ')
    answer_text = answer_text.replace('}', ' ')
    answer_text = answer_text.replace(r'\(', ' ')
    answer_text = answer_text.replace(r'\)', ' ')
    answer_text = answer_text.replace(r'(', ' ')
    answer_text = answer_text.replace(r')', ' ')

    output = _tokenize_answer_string_with_unit_prefix_and_suffix(answer_text)
    if standardize_units:
        output = _standardize_symbols_list(output, Generics.STANDARDIZE_UNITS)
    return output


def process_and_split_answer_string_exact_match(answer_text: str, ignore_case: bool, ignore_punctuation: bool, ignore_space: bool) -> list:
    if ignore_case:
        answer_text = answer_text.lower()

    if ignore_punctuation:
        # remove all punctuations in the string that can be ignored and replace with space
        for punctuation in Generics.EXACTMATCH_PUNCTUATION_SYMBOLS:
            answer_text = answer_text.replace(' ' + punctuation + ' ', ' ')
            answer_text = answer_text.replace(' ' + punctuation, ' ')
            answer_text = answer_text.replace(punctuation + ' ', ' ')
            answer_text = answer_text.replace(punctuation, ' ')

    if ignore_space:
        answer_text = _process_space_around_numbers(answer_text,
                                                    insert_space_between_number_letter=True,
                                                    remove_space_between_numbers_or_decimals_or_slash=True)
        answer_text = _standardize_symbols_string(answer_text, Generics.STANDARDIZE_SYMBOLS_EXACT_MATCH,
                                                  insert_space=True)

    # replace braces with division (for frac) or spaces
    answer_text = answer_text.replace('}{', ' / ')
    answer_text = answer_text.replace('{', ' ')
    answer_text = answer_text.replace('}', ' ')
    answer_text = answer_text.replace(r'\(', ' ')
    answer_text = answer_text.replace(r'\)', ' ')
    answer_text = answer_text.replace(r'(', ' ')
    answer_text = answer_text.replace(r')', ' ')

    # pad space before and after tokens
    for token in Generics.EXACTMATCH_TOKENS:
        answer_text = answer_text.replace(token, ' ' + token + ' ')

    # split by space
    if ignore_space:
        return list(filter(None, answer_text.split(' ')))
    else:
        return answer_text.split(' ')


def process_and_split_answer_string_numbers(answer_text: str, insert_space_standardize: bool = False) -> list:
    """
    approach: add spaces between numeric_tokens and numbers, then split into a list
    """
    answer_text = _process_space_around_numbers(answer_text,
                                                insert_space_between_number_letter=True,
                                                remove_space_between_numbers_or_decimals_or_slash=True)

    answer_text = _standardize_symbols_string(answer_text, Generics.STANDARDIZE_SYMBOLS, insert_space=insert_space_standardize)
    # answer_text = _standardize_symbols_string(answer_text, Generics.STANDARDIZE_UNITS)

    # replace braces with division (for frac) or spaces
    answer_text = answer_text.replace('}{', ' / ')
    answer_text = answer_text.replace('{', ' ')
    answer_text = answer_text.replace('}', ' ')
    answer_text = answer_text.replace(r'\(', ' ')
    answer_text = answer_text.replace(r'\)', ' ')
    answer_text = answer_text.replace(r'(', ' ')
    answer_text = answer_text.replace(r')', ' ')

    # pad space before and after tokens
    for token in Generics.NUMERIC_TOKENS:
        answer_text = answer_text.replace(token, ' ' + token + ' ')

    output = _tokenize_answer_string_with_unit_prefix_and_suffix(answer_text)
    output = _standardize_symbols_list(output, Generics.STANDARDIZE_UNITS)
    return output


def process_and_split_answer_string_algebra(answer_text: str, units: dict) -> list:
    answer_text = _standardize_symbols_string(answer_text, Generics.STANDARDIZE_SYMBOLS_EXACT_MATCH,
                                              insert_space=True)
    answer_text = _process_answer_string_algebra(answer_text)

    # Here, deal with dollar sign, convert all "\\$" to "$" first
    answer_text = answer_text.replace("\\$", "$")
    has_dollar_sign_prefix = False
    if answer_text.startswith('$'):
        answer_text = answer_text[1:]
        has_dollar_sign_prefix = True

    has_unit_suffix = False
    split_answer_space = answer_text.split(sep=' ')
    for unit in units.values():
        if split_answer_space[-1] in unit:
            answer_text = " ".join(split_answer_space[:-1])
            has_unit_suffix = True

    output = _split_answer_string_algebra(answer_text)
    if has_dollar_sign_prefix:
        output.insert(0, '$')

    if has_unit_suffix:
        output.append(split_answer_space[-1])

    return output


def preprocess_answer_string(answer_text) -> str:
    # remove enclosing white space and brackets
    # answer_text = answer_text.strip(' ()')
    answer_text = convert_escaped_space(answer_text)
    answer_text = _remove_ending_escape_char(answer_text) # if answer ends with a single escape backslash somehow
    answer_text = _remove_enclosing_brackets_and_whitespace(answer_text)  # do this once
    answer_text = _remove_escaped_brackets(answer_text)
    answer_text = _remove_br_tags_from_answer_string(answer_text)
    answer_text = _remove_left_right_tags_from_answer_string(answer_text)
    answer_text = _remove_text_tags_from_answer_text(answer_text)
    answer_text = _remove_enclosing_brackets_and_whitespace(answer_text)  # do this again

    #answer_text = _standardize_symbols_string(answer_text, Generics.STANDARDIZE_SYMBOLS)
    #answer_text = _standardize_symbols_string(answer_text, Generics.STANDARDIZE_UNITS)

    return answer_text


def insert_multiplication_token(answer_tokens: list) -> list:
    output = [answer_tokens[0]]
    for a, b in list(zip(answer_tokens, answer_tokens[1:])):
        if (is_number(a) and is_literal(b)) or (is_literal(a) and is_number(b)) or \
                (is_literal(a) and is_literal(b)) or (a == ')' and b == '(') or \
                (is_number(a) and b == '(') or (a == ')' and is_number(b)) or \
                (is_literal(a) and b == '(') or (a == ')' and is_literal(b)) or \
                (a == '}' and is_literal(b)) or (a == '}' and is_number(b)) or \
                (a == '}' and b == '(') or \
                (is_number(a) and b.startswith('\\')):
            output.append('*')
        output.append(b)

    return output


# splits a multiple-part answer string (eg, a=3, b=4) and return dict if there's equal sign and list otherwise
def split_answer_by_separators(answer_text: str, separator_custom=None) -> list:
    output = []
    separator = ";|,"
    if separator_custom:
        separator_custom = separator_custom if isinstance(separator_custom, list) else [separator_custom]
        separator = separator + "|" + "|".join(separator_custom)
    cur_LHS = ''
    for answer in list(filter(None, regex.split(separator, answer_text))):
        if '=' in answer and '<=' not in answer and '>=' not in answer:
            splits = answer.split('=', 1)
            cur_LHS = splits[0].strip()
            output.append((cur_LHS, splits[1].strip()))  # add as tuple
        else:
            output.append((cur_LHS, answer.strip()))
    return output


# converts any escaped space "\ " in string to normal space
def convert_escaped_space(answer_text: str) -> str:
    return answer_text.replace(r'\ ', ' ')


def _process_answer_string_algebra(answer_text) -> str:
    answer_text = _convert_frac_tags(answer_text)
    answer_text = _insert_multiplication_sign_before_sqrt(answer_text)
    return answer_text


def _split_answer_string_algebra(answer_text: str) -> list:
    all_symbols = "|\\".join(Generics.STANDARD_MATH_SYMBOLS)
    mathjax_symbols = "|\\".join([symbol for symbols in Generics.MATHJAX_SYMBOLS for symbol in symbols])
    regex_string = mathjax_symbols + r'|[A-Za-z]|\d*\.?\d+|[\\)]|[\\(]|[)]|[(]|[\\\\A-Za-z]+|{|}|' + "\\" + all_symbols
    #regex_string = all_symbols[2:] + r'|\d+|[\\)]|[\\(]|[)]|[(]|[\\\\A-Za-z]+|{|}|[A-Za-z]'

    # this regex splits text, num and mathjax tags and brackets
    return regex.findall(regex_string, answer_text)


def _process_space_around_numbers(answer_text: str,
                                  insert_space_between_number_letter: bool = False,
                                  remove_space_between_numbers_or_decimals_or_slash: bool = False) -> str:

    # Add a space between equals sign and number
    answer_text = regex.sub(r"(?i)(?<==)(?=\d)|(?<=\d)(?==)", r" ", answer_text)
    # Add a space between equals sign and letter
    answer_text = regex.sub(r"(?i)(?<==)(?=[a-z])|(?<=[a-z])(?==)", r" ", answer_text)

    if insert_space_between_number_letter:
        # Add a space between number and letter or letter and number attached together, eg 75thousand
        answer_text = regex.sub(r"(?i)(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)", r" ", answer_text)
        # Add a space between number and litres and other symbols, eg 48ℓ
        answer_text = regex.sub(r"(?i)(?<=\d)(?=ℓ)", r" ", answer_text)
        #answer_text = regex.sub(r"(?i)(?<=\d)(?=°)", r" ", answer_text)
        answer_text = regex.sub(r"(?i)(?<=\d)(?=₵)", r" ", answer_text)

    if remove_space_between_numbers_or_decimals_or_slash:
        answer_text = regex.sub(r"(?<=\d)(\\ )+(?=\d)|(?<=\d)(\\\\ )+(?=\d)", r"", answer_text)
        answer_text = regex.sub(r"(?<=\.)(\\ )+(?=\d)|(?<=\.)(\\\\ )+(?=\d)", r"", answer_text)
        answer_text = regex.sub(r"(?<=\d)(\\\\ )+(?=\.)", r"", answer_text)
        answer_text = regex.sub(r"(?<=\d)\s+(?=\d)|(?<=\.)\s+(?=\d)|(?<=\d)\s+(?=\.)", r"", answer_text)
        answer_text = regex.sub(r"(?<=/)(\\ )+(?=[A-Za-z])|(?<=/)(\\\\ )+(?=[A-Za-z])", r"", answer_text)
        answer_text = regex.sub(r"(?<=[A-Za-z])(\\\\ )+(?=/)|(?<=[A-Za-z])(\\ )+(?=/)", r"", answer_text)
        answer_text = regex.sub(r"(?<=/)\s+(?=[A-Za-z])|(?<=[A-Za-z])\s+(?=/)", r"", answer_text)

    return answer_text


def _tokenize_answer_string(answer_text: str) -> list:
    return list(filter(None, regex.split(" |,", answer_text)))


def _tokenize_answer_string_with_unit_prefix_and_suffix(answer_text: str) -> list:
    """
    Tokenize by splitting with _tokenize_answer_string(), but also handles dollar sign prefix and prevent tokenizing
    partial unit in text
    """
    # handle dollar sign
    has_dollar_sign_prefix = False
    if answer_text.startswith('$'):
        answer_text = answer_text[1:]
        has_dollar_sign_prefix = True

    if answer_text.startswith('\\$'):
        answer_text = answer_text[2:]
        has_dollar_sign_prefix = True

    has_unit_suffix = False
    split_answer_space = answer_text.split(sep=' ')
    for unit in Generics.UNITS.values():
        if split_answer_space[-1] in unit:
            answer_text = " ".join(split_answer_space[:-1])
            has_unit_suffix = True

    output = _tokenize_answer_string(answer_text)
    if has_dollar_sign_prefix:
        output.insert(0, '$')

    if has_unit_suffix:
        output.append(split_answer_space[-1])

    return output


def has_joined_numbers_letters(answer_text: str) -> bool:
    return regex.search(r"(?i)(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)", answer_text) is not None


def _standardize_symbols_string(answer_text: str, symbol_table: list, insert_space: bool = False) -> str:
    #[answer_text.replace(symbol, symbols[0]) for symbols in symbol_table for symbol in symbols if symbol in answer_text]
    for standard, symbols in symbol_table.items():
        for symbol in symbols:
            if symbol in answer_text:
                answer_text = answer_text.replace(symbol, " " + standard + " "
                if insert_space and standard is not "/"
                else standard)

    return answer_text


def _standardize_symbols_list(answer_text: list, symbol_table: list) -> str:
    # if symbol is found in answer_text, replace it with standard, else do nothing
    for standard, symbols in symbol_table.items():
        answer_text = [standard if token in symbols else token for token in answer_text]

    return answer_text


def _convert_frac_tags(answer_text: str) -> str:
    frac_tag_open = r"\frac{"
    while frac_tag_open in answer_text:
        index = answer_text.find(frac_tag_open)
        text_front = answer_text[:index]
        text_numerator = ""
        text_denominator = ""
        text_back = ""

        # Do Numerator
        level = 0
        text_temp = ""
        for char in answer_text[index + 6:]:
            if level < 0:
                text_temp = text_temp + char
            else:
                if char == "{":
                    level += 1
                elif char == "}":
                    level -= 1

                if level >= 0:
                    text_numerator = text_numerator + char

        # safety check. if this hits true means the formatting is incorrect then do nothing
        if text_temp[0] != "{":
            return answer_text

        # Do Denominator and Back
        level = 0
        for char in text_temp[1:]:
            if level < 0:
                text_back = text_back + char
            else:
                if char == "{":
                    level += 1
                elif char == "}":
                    level -= 1

                if level >= 0:
                    text_denominator = text_denominator + char

        answer_text = text_front + "((" + text_numerator + ")/(" + text_denominator + "))" + text_back

    return answer_text


def _remove_escaped_brackets(answer_text: str) -> str:
    return answer_text.replace("\\(", "").replace("\\)", "")


def _insert_multiplication_sign_before_sqrt(answer_text: str) -> str:
    mult_sign = '*'
    # insert sign if char before sqrt is alphanumeric
    index_before_sqrt = [m.start()-1 for m in regex.finditer(r"\\sqrt{", answer_text) if m.start() > 0]
    for index in reversed(index_before_sqrt):
        char = answer_text[index]
        if char.isspace() and index > 0:
            char = answer_text[index-1]
        if char.isalnum() or char == ")" or char == "}":
            answer_text = __insert_str_in_str_at_idx(answer_text, mult_sign, index+1)

    return answer_text


def __insert_str_in_str_at_idx(string: str, string_to_add: str, index: int) -> str:
    return string[:index] + string_to_add + string[index:]


def remove_comma(string: str):
    return string.replace(',', '')

