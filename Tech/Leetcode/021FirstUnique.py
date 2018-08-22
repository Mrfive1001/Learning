class Solution(object):
    def firstUniqChar(self, s):
        """
        :type s: str
        :rtype: int
        """
        repeat = []
        for i in s:
            if i in repeat:
                continue
            else:
                if s.count(i) == 1:
                    return s.index(i)
                else:
                    repeat.append(i)
        return -1

test = Solution()
print(test.firstUniqChar('leetcode'))