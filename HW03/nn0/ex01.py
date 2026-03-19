# 1. 多項式微積分小考（測試 Value 與 backward）

from nn0 import Value
# 1. 建立一個測試用的 x 變數，並給予初始值 2.0
x = Value(2.0)

# 2. 定義我們的多項式方程式：y = 3x^2 + 2x
# 注意：這裡的運算會自動觸發 Value 類別裡面的 __mul__, __pow__, __add__ 等方法
y = 3 * x**2 + 2 * x

# 3. 印出前向傳播（Forward Pass）算出來的 y 值
print("--- 前向計算 ---")
print(f"當 x = {x.data} 時，y 的計算結果為：{y.data}")

# 4. 啟動反向傳播（Backward Pass），程式會沿著運算圖往回計算梯度
y.backward()

# 5. 印出算出來的梯度（微分結果）
print("\n--- 反向傳播 ---")
print(f"程式自動算出的 x 梯度（斜率 dy/dx）為：{x.grad}")