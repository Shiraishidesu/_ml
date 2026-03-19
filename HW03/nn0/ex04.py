# 4. 手刻簡單線性迴歸找規律（單純測試 Adam 優化器如何更新參數）

from nn0 import Value, Adam

# 1. 準備「教材」：一組符合 y = 2x + 1 規律的資料
xs = [1.0, 2.0, 3.0, 4.0, 5.0]
ys = [3.0, 5.0, 7.0, 9.0, 11.0]

# 2. 模型初始化：給定兩個一無所知的參數 w 和 b (把它們當作 Value 變數)
w = Value(0.0)
b = Value(0.0)

# 3. 請出教練：建立 Adam 優化器，負責訓練 w 和 b。我們把學習率 (lr) 設為 0.1 讓它學快一點
optimizer = Adam([w, b], lr=0.1)

print("=== 訓練開始 ===")
print(f"初始猜測方程式: y = {w.data:.2f}x + {b.data:.2f}")

# 4. 開始瘋狂做題（訓練迴圈 Training Loop）
epochs = 100  # 總共讓它把這份教材讀 100 遍

for step in range(epochs):
    # 每次重讀前，先準備一個計分板來記錄總誤差
    total_loss = Value(0.0)
    
    # 把 5 筆資料拿出來做測驗
    for x_val, y_val in zip(xs, ys):
        # [步驟 A：猜答案 (Forward)]
        y_pred = w * x_val + b
        
        # [步驟 B：算誤差 (Loss)] -> 使用均方誤差 (預測值 - 真實值)^2
        loss = (y_pred - y_val)**2
        total_loss = total_loss + loss
        
    # 計算平均誤差
    avg_loss = total_loss / len(xs)
    
    # [步驟 C：找戰犯 (Backward)] -> 自動算出 w 和 b 該負多少責任 (梯度)
    avg_loss.backward()
    
    # [步驟 D：訂正 (Update)] -> Adam 根據梯度，微調 w 和 b 的數值，並清空梯度
    optimizer.step()
    
    # 每 20 次印出一份成績單
    if (step + 1) % 20 == 0:
        print(f"第 {step+1:3d} 次練習 | 誤差(Loss): {avg_loss.data:6.4f} | 模型參數: w={w.data:.4f}, b={b.data:.4f}")

print("\n=== 訓練結束 ===")
print(f"最終模型學到的方程式: y = {w.data:.2f}x + {b.data:.2f}")
print("我們心中的標準答案:   y = 2.00x + 1.00")