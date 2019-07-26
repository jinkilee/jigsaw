import re

# FIXME: regex pattern
number = re.compile(r'(\s(\-|)\d+)')

special_chars = [
    ';',
    ':',
    '"',
    '\'',
    '(',
    ')',
    '{',
    '}',
    '\.',
    ',',
    '\?',
    '/',
    '!',
    '@',
    '#',
    '$',
    '%',
    '^',
    '&',
    '\*',
    '_',
    '\-',
    '\+',
    '=',
    '\r\n',
    '\n',
    '\\\\',
    '`',
    '>',
    '<',
    '~',
]
sp_pattern = '[{}]'.format('|'.join(special_chars))
spchar = re.compile(sp_pattern)

regex_pattern_list = [
	(number, ' [NUMBER] '),
	(spchar, ' '),
]
