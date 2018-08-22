class Solution(object):
    def findWords(self, words):
        """
        :type words: List[str]
        :rtype: List[str]
        """
        list = ['qwertyuiop','asdfghjkl','zxcvbnm']
        result = []
        for word in words:
            for i in list:
                if self.isinlist(word,i):
                    result.append(word)
                else:
                    continue
        return result

    def isinlist(self,word,list):
        word = word.lower()
        for j in word:
            if j in list:
                continue
            else:
                return False
        return True


test = Solution()
print(test.findWords(["shdfkj",'sdfsdf','ewrqg','zhou','ASzc','ASds']))