# Data Analysis

## Kmeas

```flow
st=>start: 开始
e=>end: 结束
div=>operation: 划分训练集和测试集
pick_cent=>operation: 选取质心
calc_dist=>operation: 计算各数据点与质心的欧氏距离
并划分数据到距离最近的质心
sse_cond=>condition: 误差平方和SSE收敛
train_io=>inputoutput: 输出K个划分集
test=>operation: 使用测试集测试划分
test_cond=>condition: 满足测试条件
test_pass=>inputoutput: 通过测试数据
test_exc=>inputoutput: 判定异常数据

st->div->pick_cent->calc_dist->sse_cond
sse_cond(yes)->train_io
sse_cond(no)->calc_dist
train_io->test->test_cond
test_cond(yes)->test_pass
test_cond(no)->test_exc
```
