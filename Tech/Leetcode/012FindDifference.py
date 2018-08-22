class Solution(object):
    def findTheDifference(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        s1 = list(s)
        t1 = list(t)
        for i in s1:
            t1.pop(t1.index(i))
        return t1[0]

test = Solution()
print(test.findTheDifference('abcd','abcde'))