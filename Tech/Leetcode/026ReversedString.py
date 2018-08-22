class Solution(object):
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        a = s.split(' ')
        result = ''
        for j, i in enumerate(a):
            a[j] = i[::-1]
        result = ' '.join(a)
        return result
test = Solution()
print(test.reverseWords('let,sdf sdf sdf'))