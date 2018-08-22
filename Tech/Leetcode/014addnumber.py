import functools
class Solution(object):
    def addDigits(self, num):
        """
        :type num: int
        :rtype: int
        """
        while num//10:
            a = list(str(num))
            num = functools.reduce((lambda x,y:x+y),map((lambda x:int(x)),a))
        return num
test = Solution()
print(test.addDigits(23))