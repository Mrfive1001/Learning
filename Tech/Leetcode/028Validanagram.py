class Solution(object):
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        m = sorted(s)
        l = sorted(t)
        return True if m==l else False

test = Solution()
print(test.isAnagram('leetcode','lteecode'))