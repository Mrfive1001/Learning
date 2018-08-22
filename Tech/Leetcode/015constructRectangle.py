class Solution(object):
    def constructRectangle(self, area):
        """
        :type num: int
        :rtype: List[int]
        """
        num = area ** 0.5
        if num ==int(num):
            result = [int(num), int(num)]
        else:
            num = int(num) + 1
            while area%num:
                num += 1
            result = [num,int(area/num)]
        return result
test = Solution()
print(test.constructRectangle(1))