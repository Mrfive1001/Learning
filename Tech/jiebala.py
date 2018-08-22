import jieba
import jieba.posseg as psg

# 分词模式
## 精确模式
s = u'我想和女朋友一起去北京故宫博物院参观和闲逛。'
cut = jieba.cut(s)
print(cut)
print(','.join(cut))
## 全模式 把文本分成尽可能多的词
print(','.join(jieba.cut(s, cut_all=True)))
## 搜索引擎模式,可以发现词典中没有的词
print(','.join(jieba.cut_for_search(s)))
# 获取词性 使用posseg中切分
print([(x.word, x.flag) for x in psg.cut(s)])
# 并行分词 加速分词效率
# 获取出现频率top n的词