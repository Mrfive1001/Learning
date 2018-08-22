class Solution(object):
    def arrayPairSum(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        return sum(sorted(nums)[::2])
test = Solution()
print(test.arrayPairSum([1, 3, 4, 5, 2, 4]))