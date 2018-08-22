class Solution(object):
    def findContentChildren(self, g, s):
        """
        :type g: List[int]
        :type s: List[int]
        :rtype: int
        """
        g.sort()
        s.sort()
        i = 0
        for j in s:
            if not g:
                break
            else:
                if j >= g[0]:
                    g.pop(0)
                    i += 1
        return i

test = Solution()
print(test.findContentChildren([1,2,3],[1,1]))