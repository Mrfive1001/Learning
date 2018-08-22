class Solution(object):
    def findMaxConsecutiveOnes(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        count = 0
        max = 0
        for num in nums:
            if num:
                count+=1
                if count>max:
                    max=count
            else:
                count=0
        return max
test = Solution()
print(test.findMaxConsecutiveOnes([1,1,0,1]))