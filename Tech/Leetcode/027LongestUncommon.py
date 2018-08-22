class Solution(object):
    def findLUSlength(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: int
        """
        # if len(a) != len(b):
        #     return max((len(a), len(b)))
        # elif a == b:
        #     return -1
        # else:
        #     return len(a)
        return (max((len(a),len(b))) if len(a) != len(b) else (-1 if a==b else len(a)))

test = Solution()
print(test.findLUSlength('leetcode','sdfsad'))