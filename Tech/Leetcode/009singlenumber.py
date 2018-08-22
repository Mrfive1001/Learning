class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # for i,num in enumerate(nums):
        #     a = nums[:]
        #     print(a)
        #     a.pop(i)
        #     if num not in a:
        #         return num
        return 2*sum(set(nums))-sum(nums)
test = Solution()
print(test.singleNumber([1,1,2]))