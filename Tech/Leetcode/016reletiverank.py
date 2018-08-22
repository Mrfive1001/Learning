class Solution(object):
    def findRelativeRanks(self, nums):
        """
        :type nums: List[int]
        :rtype: List[str]
        """
        num = sorted(nums,reverse=True)
        a = [str(num.index(x)+1) for x in nums]
        a[a.index('1')] = 'Gold Medal'
        a[a.index('2')] = 'Silver Medal'
        a[a.index('3')] = 'Bronze Medal'
        return a
test = Solution()
print(test.findRelativeRanks([10,3,8,9,4]))