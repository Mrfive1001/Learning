class Solution(object):
    def getSum(self, a, b):
        """
        :type a: int
        :type b: int
        :rtype: int
        """
        c = list((a,b))
        return sum(c)

test = Solution()
print(test.getSum(1,2))