text = u"""

"""
newText = ''
for t in text:
    if t == ' ' or t == '\n':
        continue
    elif 58830 <= ord(t) <= 58840:
        continue
    elif 65313 <= ord(t) <= 65338:
        t = chr(ord(t) - (65313 - ord('A')))
    elif 65345 <= ord(t) <= 65370:
        t = chr(ord(t) - (65345 - ord('a')))
    elif 65296 <= ord(t) <= 65305:
        t = chr(ord(t) - (65296 - ord('0')))
    newText = newText + t
print(newText)
