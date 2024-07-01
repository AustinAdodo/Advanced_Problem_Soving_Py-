# Press the green button in the gutter to run the script.
# from Puzzles import pickingNumbers
import socket
import collections
from typing import List

from Puzzles import firstMissingPositive


# letters = []
# characters = [chr(c) for c in range(ord('a'), ord('z') + 1)]
# for letter_code in range(ord('A'), ord('Z') + 1):
#     letter = chr(letter_code)
#     letters.append(letter)
# Now the 'letters' list contains letters A to Z

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


def lengthOfLongestSubstring(s: str) -> int:
    """
             sliding Window has been utilised.
             """
    if len(s) > 1 and all(char == s[0] for char in s):
        return 1
    if s == "" or not s:
        return 0
    char_index_map = {}
    max_length = 0
    start_index = 0
    for end_index, char in enumerate(s):
        if char in char_index_map and char_index_map[char] >= start_index:
            start_index = char_index_map[char] + 1
        char_index_map[char] = end_index
        max_length = max(max_length, end_index - start_index + 1)
    return max_length


def are_indexes_adjacent(arr_input: []) -> bool:
    new_arr = [x for x in arr_input if x >= 1]
    element2 = new_arr[1]
    element1 = new_arr[0]
    index1 = arr_input.index(element1)
    index2 = arr_input.index(element2)
    if element1 != element2:
        return abs(index1 - index2) == 1
    second_occurrence_index = arr_input.index(element2, index1 + 1)
    return abs(second_occurrence_index - index2) == 1


def trap(height: list[int]) -> int:
    """
             Approach:  len(height) < 3 return 0 because it takes at least
             2 walls and a pit to store water.
             while (a >= 1 for a in height) >= 2 sum of the walls with height > 1 when counted are more than 2

             Ensure heights of 1 unit or more are more than 1 in number and if 2 are not consecutive
    """
    result = wall = temp = 0
    if all(x == height[0] for x in height):
        return 0
    segregated_heights = [a for a in height if a >= 1]
    heights_more_than_or_eq_1unit = len(segregated_heights)
    min_height_greater_than_zero = min(segregated_heights)
    if len(height) < 3:
        return 0
    # handle all levels
    while heights_more_than_or_eq_1unit >= 2:
        wall = temp = 0
        for i, e in enumerate(height):
            if e > 0 and wall == 0:
                wall = min_height_greater_than_zero
                continue
            if wall >= 1 and e == 0:
                temp += min_height_greater_than_zero
            if wall >= 1 and e > 0:
                result += temp
                temp = 0
        # move to the next level using the smallest number in the array as reference
        height = [a - min_height_greater_than_zero if a > 0 else 0 for a in height]
        # proper heights
        segregated_heights = [a for a in height if a >= 1]
        heights_more_than_or_eq_1unit = len(segregated_heights)
        min_height_greater_than_zero = min(segregated_heights) if len(segregated_heights) > 1 else 0
        # heights of 1 unit or more are more than 1 in number and if 2 are not consecutive
        if heights_more_than_or_eq_1unit == 2 and are_indexes_adjacent(height):
            break
    return result


def trap2(height: List[int]) -> int:
    if len(height) <= 2:
        return 0

    ans = 0
    i, j = 1, len(height) - 1
    lmax, rmax = height[0], height[-1]

    while i <= j:
        # Update left and right maximum for the current positions
        if height[i] > lmax:
            lmax = height[i]
        if height[j] > rmax:
            rmax = height[j]

        # Fill water up to lmax level for index i and move i to the right
        if lmax <= rmax:
            ans += lmax - height[i]
            i += 1
        # Fill water up to rmax level for index j and move j to the left
        else:
            ans += rmax - height[j]
            j -= 1

    return ans


def get_smallest_substring(str1, str2):
    # I need to maintain time and space complexity On
    # using sliding rail is my best option
    from collections import Counter
    required = Counter(str2)
    required_chars = len(required)

    left = 0
    right = 0
    formed = 0
    window_counts = {}
    min_length = float('inf')
    min_window = (0, 0)

    while right < len(str1):
        char = str1[right]
        window_counts[char] = window_counts.get(char, 0) + 1
        if char in required and window_counts[char] == required[char]:
            formed += 1
        while left <= right and formed == required_chars:
            char = str1[left]
            if (right - left + 1) < min_length:
                min_length = right - left + 1
                min_window = (left, right)
            window_counts[char] -= 1
            if char in required and window_counts[char] < required[char]:
                formed -= 1
            left += 1
        right += 1

    if min_length == float('inf'):
        return ""
    else:
        return str1[min_window[0]:min_window[1] + 1]


def get_smallest_substring2(str1, str2):
    if len(str2) > len(str1):
        return ""
    checker = dict.fromkeys(str2, False)
    arr = []
    string_checker = ""
    compared_length = len(str2)
    l = len(str1)
    start = 0
    while l - start >= compared_length:
        for char in range(start, l):
            if all(value is True for value in checker.values()):
                break
            else:
                if str1[char] in checker and checker[str1[char]] is False:
                    checker[str1[char]] = True
                    string_checker += str1[char]
                else:
                    string_checker += str1[char]
        string_checker.strip()
        if all(element in string_checker for element in str2):
            arr.append(string_checker)
        checker = dict.fromkeys(str2, False)
        string_checker = ""
        start += 1

    min_length = min(len(x) for x in arr)
    return [x for x in arr if len(x) == min_length][0]


def integer_from_string(input_string):
    input_string = input_string.strip()

    if not input_string:
        return 0

    sign = 1
    start = 0

    if input_string[0] == '-':
        sign = -1
        start = 1
    elif input_string[0] == '+':
        start = 1

    num_str = ''
    for char in input_string[start:]:
        if char.isdigit():
            num_str += char
        else:
            break

    if not num_str:
        return 0

    result = sign * int(num_str)

    # Ensure the result is within the 32-bit signed integer range
    if result < -2147483648:
        return -2147483648
    if result > 2147483647:
        return 2147483647

    return result


if __name__ == '__main__':
    # print(get_smallest_substring("ADOBECODEBANC", "ABC"))  # Output: "BANC"
    # print(get_smallest_substring("geeksforgeeks", "ork"))  # Output: "ksfor"
    # print(get_smallest_substring("a", "a"))  # Output: "a"
    # print(get_smallest_substring("a", "aa"))  # Output: ""

    print(get_smallest_substring2("ADOBECODEBANC", "ABC"))  # Output: "BANC"
    print(get_smallest_substring2("geeksforgeeks", "ork"))  # Output: "ksfor"
    print(get_smallest_substring2("a", "a"))  # Output: "a"
    print(get_smallest_substring2("a", "aa"))  # Output: ""
