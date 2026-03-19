# 3. 訊號音量控制器實驗

from nn0 import Value
from nn0 import rmsnorm
# 假設這是一層神經網路剛剛算出來的結果
# 這裡面的數值差距極大，有的非常大 (50)，有的非常小 (0.1) 甚至為負 (-20)
raw_signals = [Value(0.1), Value(50.0), Value(-20.0), Value(0.005)]

print("=== 處理前：原始訊號 (音量失控) ===")
for i, s in enumerate(raw_signals):
    # 格式化輸出，對齊小數點方便觀察
    print(f"通道 {i+1} 的數值: {s.data:8.4f}")

# 通過 RMSNorm 音量控制器
normalized_signals = rmsnorm(raw_signals)

print("\n=== 處理後：RMSNorm 轉換 (音量被標準化) ===")
for i, s in enumerate(normalized_signals):
    print(f"通道 {i+1} 的數值: {s.data:8.4f}")