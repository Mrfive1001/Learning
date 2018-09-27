# Google Python编程规范
## import的使用
## Exception的使用
* 一般是指打破通常循环来解决错误或者其他额外情况的语句。
* raise Myerror('message')
## Global variables
* 尽量不要用
* 使用的时候全部大写并且使用_相连接
## Python特色表达
* 简单的情况可以使用，不能使用在复杂情况
* 列表生成器
* 条件表达式
* 匿名函数
## Default value
* 非常好用的方法
* 可以使用默认的value或者None
* 使用list或者dict的时候可能会出现错误
## Property
* [Property的用法](https://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000/00143186781871161bc8d6497004764b398401a401d4cce000)
* 可以加一层中间函数的限制的调用
## Comments and Docstring
## Naming
module_name, package_name, ClassName, method_name, ExceptionName, function_name, GLOBAL_CONSTANT_NAME, global_var_name, instance_var_name, function_parameter_name, local_var_name. 
# Reference
[谷歌官方文档](https://github.com/google/styleguide/blob/gh-pages/pyguide.md)