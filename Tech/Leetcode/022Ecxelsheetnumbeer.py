class Solution(object):
    def titleToNumber(self, s):
        """
        :type s: str
        :rtype: int
        """
        result = 0
        for value in s:
            result = result * 26 + ord(value)-64
        return result
test = Solution()
print(test.titleToNumber('AA'))