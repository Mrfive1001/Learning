class Solution(object):
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        return False if len(nums) == len(set(nums)) else True

test = Solution()
print(test.containsDuplicate([1,3,3,4,6]))