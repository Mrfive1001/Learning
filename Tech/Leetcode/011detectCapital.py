class Solution(object):
    def detectCapitalUse(self, word):
        """
        :type word: str
        :rtype: bool
        """
        if word.islower() or word.isupper():
            return True
        elif word.capitalize() == word:
            return True
        else:
            return False

test = Solution()
print(test.detectCapitalUse('USA'))