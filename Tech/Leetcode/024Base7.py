class Solution(object):
    def convertToBase7(self, num):
        """
        :type num: int
        :rtype: str
        """
        n = 7
        times = 0
        symbol = 1
        result = 0
        if num < 0:
            num = -num
            symbol = -1
        while num >= 7:
            result += num % n * (10**times)
            num = num // n
            times += 1
        result = symbol * (num * (10**times) + result)
        return str(result)



test = Solution()
print(test.convertToBase7(-200))

