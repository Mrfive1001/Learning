class Solution(object):
    def canConstruct(self, ransomNote, magazine):
        """
        :type ransomNote: str
        :type magazine: str
        :rtype: bool
        """
        a = sorted(list(ransomNote))
        b = sorted(list(magazine))
        for i in a:
            if i not in b:
                return False
            else:
                b.pop(b.index(i))
        return True
test = Solution()
print(test.canConstruct('dsaf','dsf'))