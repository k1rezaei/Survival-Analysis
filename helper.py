import numpy as np
import re


def isDate(date):
    if isinstance(date, str):
        if re.search(r"^\d+/\d*/\d*$", date):
            return True
        return False
    else:
        return False


def isNumDate(num):
    if isinstance(num, int):
        if num > 13000000:
            return True
        return False
    else:
        return False


def getDate(str):
    if not isDate(str):
        if isNumDate(str):
            year = int(str / 10000) - 1300
            month = int((str % 10000) / 100)
            day = int(str % 100)
            if month > 12:
                month = 12
            if day > 31:
                day = 31
            return [year, month, day]
        else:
            return [0, 0, 0]
    date = str.split("/")

    if int(date[0]) > 1300:
        date[0] = int(date[0]) - 1300
    else:
        date[0] = int(date[0])

    if date[1] != '':
        if 0 < int(date[1]) < 13:
            date[1] = int(date[1]) - 1
        else:
            date[1] = 0
    else:
        date[1] = 0

    if date[2] != '':
        if 0 < int(date[2]) < 32:
            date[2] = int(date[2]) - 1
        else:
            date[2] = 0
    else:
        date[2] = 0

    return date


def getNormalDate(str1, str2):
    date1 = getDate(str1)
    if sum(date1) == 0:
        return 0
    date2 = getDate(str2)
    if sum(date2) == 0:
        return 0
    diff = np.array(date1) - np.array(date2)

    return diff[0] * 365 + diff[1] * 30 + diff[2]


def setDate(str1, str2):
    diff = getNormalDate(str1, str2)
    if diff == 0:
        diff = 10000
    return diff
