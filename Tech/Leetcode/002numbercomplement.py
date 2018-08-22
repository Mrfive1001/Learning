class Solution(object):
    def findComplement(self, num):
        """
        :type num: int
        :rtype: int
        """
        temp = 2**(len(bin(num))-2)-1
        return num^temp
test = Solution()
print(test.findComplement(1))