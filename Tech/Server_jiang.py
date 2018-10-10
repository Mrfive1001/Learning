'''
Server酱来通过微信提醒
'''
import requests
url = 'https://sc.ftqq.com/SCU7896Td21f33141f474e9dfec81d7da26daacc5900906769c16.send'

text = 'Message!'
desp = '服务器炸啦哈哈哈哈'
data = {'text': text , 'desp':desp}

req = requests.post(url,data=data)