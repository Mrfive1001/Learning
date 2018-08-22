class Solution(object):
    def fizzBuzz(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        list = []
        for i in range(1,n+1):
            if i%3 == 0:
                if i%5 ==0:
                    list.append("FizzBuzz")
                else:
                    list.append("Fizz")
            elif i%5 == 0:
                list.append("Buzz")
            else:
                list.append(str(i))
        return list

test = Solution()
print(test.fizzBuzz(10))