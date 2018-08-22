class Solution(object):
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        bad = []
        for i in nums:
            if i in bad:
                continue
            elif nums.count(i) > (len(nums)/2.0):
                return i
            else:
                bad.append(i)
        return None

test = Solution()
print(test.majorityElement([12, 12, 55, 22, 12]))