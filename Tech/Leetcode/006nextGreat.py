class Solution(object):
    def nextGreaterElement(self, findNums, nums):
        """
        :type findNums: List[int]
        :type nums: List[int]
        :rtype: List[int]
        """
        a = []
        for i in findNums:
            num = nums.index(i)
            if num == len(nums)-1:
                a.append(-1)
            else:
                for j in nums[num+1:]:
                    if j>i:
                        a.append(j)
                        break
                    if j==nums[-1]:
                        a.append(-1)
        return a
test = Solution()
print(test.nextGreaterElement([4,1,2],[1,3,4,2]))
