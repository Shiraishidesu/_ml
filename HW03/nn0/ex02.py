# 2. 極端數值防爆實驗（測試 softmax 函數穩定性）

import math

# 為了對比，我們先手刻一個「沒有防護機制」的天真版 Softmax
def naive_softmax(logits):
    exps = [math.exp(x) for x in logits]  # 直接把數字丟進指數函數
    total = sum(exps)
    return [e / total for e in exps]

# 假設模型在某次計算中，輸出了三個極大的預測分數（Logits）
extreme_values = [1000.0, 1001.0, 1002.0]

print("=== 實驗一：天真版 Softmax (準備迎接爆炸) ===")
try:
    print("開始計算天真版...")
    probs = naive_softmax(extreme_values)
    print("計算結果：", probs)
except OverflowError as e:
    print(f"💥 發生錯誤 (OverflowError)：{e}")
    print("原因：math.exp(1000) 的結果太大了，直接撐爆了電腦的記憶體上限！\n")


print("=== 實驗二：nn0.py 裡的數值穩定版 Softmax ===")
# 現在我們使用 nn0.py 裡面的 Value 類別與 softmax 函式
# 注意：這裡的 softmax 是呼叫你 nn0.py 裡面寫好的那個
from nn0 import Value, softmax  # 如果寫在同一個檔案，這行可省略

logits_value = [Value(1000.0), Value(1001.0), Value(1002.0)]
safe_probs = softmax(logits_value)

print("計算成功！轉換後的機率分佈為：")
for i, p in enumerate(safe_probs):
    print(f"第 {i+1} 個選項的機率: {p.data * 100:.2f}%")