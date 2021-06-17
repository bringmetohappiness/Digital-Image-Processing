a = '0100 0000 0000 0000 0000 0000 0000 0000'

def str2Q1_31(string):
    string = string.replace(' ', '')
    result = -1 if string[0] == '1' else 0
    string = string[1:]
    for i, c in enumerate(string):
        result += int(c) * pow(2, -(i + 1))
    return result


print(str2Q1_31(a))
