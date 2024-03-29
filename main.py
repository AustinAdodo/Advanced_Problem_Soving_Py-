# Press the green button in the gutter to run the script.
# from Puzzles import pickingNumbers
import socket

# a-bC-dEf=ghIj!!
# j-Ih-gfE=dCba!!
#
# ab-cd
# dc-ba
import collections

from Puzzles import firstMissingPositive


# letters = []
# characters = [chr(c) for c in range(ord('a'), ord('z') + 1)]
# for letter_code in range(ord('A'), ord('Z') + 1):
#     letter = chr(letter_code)
#     letters.append(letter)
# Now the 'letters' list contains letters A to Z

# Array Combination
def ArrayMergeDifferentLengths(self, nums1: list[int], nums2: list[int]) -> list[int]:
    result = nums1 + nums2
    res = sorted(result)
    return res


# median of two combined arrays of equal length.
def findMedianSortedArrays(self, nums1: list[int], nums2: list[int]) -> float:
    # the two arrays are of different sizes m,n.
    result = 0.0
    num3 = self.ArrayMergeDifferentLengths(nums1, nums2)
    # if num3 is even
    if len(num3) % 2 == 0:
        temp = int(len(num3) / 2) - 1
        temp2 = temp + 1
        result = float((num3[temp] + num3[temp2]) / 2)
    if len(num3) % 2 != 0:
        temp = int((len(num3) - 1) / 2)
        result = float(num3[temp])
    return result


def weightedUniformStrings(s, queries):
    #  Write your code here
    result = [1, 2]
    return result


# valid brackets. Determining the validity of opening and closing brackets.
def isValid(self, s):
    opcl = dict(('()', '[]', '{}'))
    stack = []
    # Traverse each character in input string...
    for idx in s:
        # If open parentheses are present, append it to stack...
        if idx in '([{':
            stack.append(idx)
        # If the character is closing parentheses, check that the same type opening parentheses is being pushed to
        # the stack or not... If not, we need to return false...
        elif len(stack) == 0 or idx != opcl[stack.pop()]:
            return False
    # At last, we check if the stack is empty or not... If the stack is empty it means every opened parenthesis is
    # being closed, and we can return true, otherwise we return false...
    return len(stack) == 0


def hashing():
    n = int(input())
    integer_list = map(int, input().split())
    print(hash(tuple(integer_list)))


def complex_Array_of_nines():
    result = []
    a = [9, 1, 2, 3, 4, 5, 6, 7, 8]
    b = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    c = [8, 9, 7, 5, 6, 4, 3, 2, 1]
    return result


def form(s="", f=""):
    lis = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o",
           "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
    j = 0
    res = ""
    for i in range(len(f)):
        if f[i] == "X" or f[i].isupper():
            res += s[j].upper()
        if f[i] == "x" or f[i].islower():
            res += s[j].lower()
        elif not (f[i] == "x" or f[i] == "X"):
            res += f[i]
            j += -1
        j += 1
    return res


def ToArray(string):
    list1 = []
    list1[:0] = string
    return list1


def divide_len_string_with_num(s=""):
    _input = s
    count = 0
    for i in range(len(_input)):
        if _input.isdigit():
            count += 1
    # get count and divide the length of string with numbers
    return round(len(_input) / count)


def camel_Cases(s: str):
    variable_name = s
    a = ToArray(variable_name)
    concat = ""
    result = list(map(lambda x: x.capitalize() if a[a.index(x) - 1] == "_" else x, a))
    print(concat.join(result).replace("_", ""))


# for i, (key, value) in enumerate(my_dict.items()):
# all_equal = all(num[1] == filtered_dic[1] for num in filtered_dic.items())
# tmp = dict(filter(lambda x: x[1] == 1, dic.items()))

def maxEqualFreq(nums1: list[int]) -> int:
    cnt, freq, maxF, res = collections.defaultdict(int), collections.defaultdict(int), 0, 0
    for i, num in enumerate(nums1):
        cnt[num] += 1
        freq[cnt[num] - 1] -= 1
        freq[cnt[num]] += 1
        maxF = max(maxF, cnt[num])
        if maxF * freq[maxF] == i or (maxF - 1) * (freq[maxF - 1] + 1) == i or maxF == 1:
            res = i + 1
    return res


def isPowerOfTwo(self, n: int) -> bool:
    if n <= 0:
        return False
    return n & (n - 1) == 0


# 2D array sorting
def two_d_arr_sort(a):
    answer = []
    ans1 = []
    sorted_list = [[]]
    sorted_list = sorted(a, key=lambda x: x[1])
    ans1 = (item[1] for item in sorted_list)
    sorted_list2 = list(set(ans1))
    sorted_list2.sort()
    sorted_list = list(filter(lambda x: x[1] == sorted_list2[1], sorted_list))
    result = sorted(sorted_list, key=lambda x: x[0])
    for item in result:
        answer.append(item[0])
    return answer
    # return sorted_list2


def Kvp():
    student_marks = {}
    query_name = ""
    n = int(input())
    for i in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
        query_name = input()
    v = student_marks[query_name]
    print('{:.2f}'.format(sum(v) / len(v)))


def minion_game(s):
    vowels = "AEIOU"
    stuart_score = 0
    kevin_score = 0
    for i in range(len(s)):
        if s[i] in vowels:
            kevin_score += len(s) - i
        else:
            stuart_score += len(s) - i
    if stuart_score > kevin_score:
        print("Stuart", stuart_score)
    elif kevin_score > stuart_score:
        print("Kevin", kevin_score)
    else:
        print("Draw")


# result.append(i if i not in nums and i <= len(nums) else 0)
def findDisappearedNumbers(nums):
    res = []
    for x in nums:
        if nums[abs(x) - 1] > 0:
            nums[abs(x) - 1] *= -1
    for i, x in enumerate(nums):
        if x > 0:
            res.append(i + 1)
    return res


def ping_public_ip():
    # port =80
    port = 443
    ip = "35.168.80.219"
    try:
        socket.create_connection((ip, port))
        print("Server is reachable")
    except ConnectionError:
        print("Server is not reachable")


def first_non_repeating_letter(s: str) -> str:
    if len(s) == 1:
        return s
    if len(s) == 0 or s == "":
        return ""
    s_lower = s.lower()

    char_count = {}

    for char in s_lower:
        if char in char_count:
            char_count[char] += 1
        else:
            char_count[char] = 1

    for char in s:
        if char_count[s_lower[s.index(char)]] == 1:
            return char

            # Return an empty string if all characters are repeated
    return ""


def all_char_unique(s: str) -> bool:
    seen_chars = set()
    for char in s:
        if char in seen_chars:
            return False
        seen_chars.add(char)
    return True


def is_convergent_on(compare: str, existing: dict) -> bool:
    sorted_dict = dict(sorted(existing.items(), key=lambda item: len(item[0]), reverse=True))
    return True if len(compare) < len(list(sorted_dict.keys())[0]) else False


def lengthOfLongestSubstring(s: str) -> int:
    dict1 = collections.defaultdict(int)
    s1 = [str(i) for i in s]
    if all(char == s1[0] for char in s1):
        return 1
    j = 0
    for i in range(len(s)):
        for j in range(i, len(s)):
            substring = s[i:j + 1]
            if all_char_unique(substring):
                dict1[substring] = dict1.get(substring, 0) + 1
            # if i > len(s) / 2 and is_convergent_on(s[i:j + 1], dict1):
            #     break

        # filtered_dict = dict(filter(lambda x: x[1] == 1, dict1.items()))
    sorted_dict = dict(sorted(dict1.items(), key=lambda item: len(item[0])))
    count = 0 if len(list(sorted_dict.items())) == 0 else len(list(sorted_dict.keys())[-1]) \
        if len(list(sorted_dict.keys())) > 1 else len(list(sorted_dict.keys())[0])
    return count


if __name__ == '__main__':
    print(lengthOfLongestSubstring('hijklmnopqrstuvwxyzABCDEFGHIJKL'
                                   'MNOPQRSTUVWXYZ0123456789hijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123'
                                   '456789hijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789hijklmnopqrstuvwxyz'
                                   'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789hijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ01'
                                   '23456789hijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'))

# a = [chr(i) for i in range(ord("A"), ord("Z") + 1)]
