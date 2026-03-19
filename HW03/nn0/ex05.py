# 5. 迷你外星語預測機 (整合測試 gd 訓練迴圈)
from nn0 import Value, Adam, gd, softmax

# --- 步驟 A：建立一個超級簡化的迷你模型 ---
class MiniModel:
    def __init__(self):
        # 為了配合 gd 函式的規格，我們必須告訴它模型的長度與層數
        self.block_size = 20
        self.n_layer = 1
        
        # 我們的外星語字典只有 3 個字：0, 1, 2
        # 我們建立一個 3x3 的「記憶矩陣」。把它想像成：W[看到什麼字][猜下一個字的分數]
        # 一開始模型什麼都不懂，所以裡面全是 Value(0.0)
        self.W = [[Value(0.0) for _ in range(3)] for _ in range(3)]

    def __call__(self, token_id, pos_id, keys, values):
        # 前向傳播 (Forward)：當看到某個字 (token_id)，就直接把「記憶矩陣」裡對應的那排分數交出去
        return self.W[token_id]

    def get_parameters(self):
        # 把矩陣裡所有的 Value 收集起來，交給 Adam 教練去訓練
        return [w for row in self.W for w in row]


# --- 步驟 B：準備教材與初始化 ---
# 這是我們要模型死背下來的外星語規律 (0接1，1接2，2接0)
alien_language = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]

model = MiniModel()
# 呼叫 Adam 優化器，學習率設大一點 (0.5) 讓它學快點
optimizer = Adam(model.get_parameters(), lr=0.5)

print("=== 語言模型訓練開始 ===")
num_steps = 50

# --- 步驟 C：啟動訓練迴圈 ---
for step in range(num_steps):
    # 呼叫 nn0.py 裡面的 gd 函式，它會自動幫我們做完：
    # 算 Logits -> 轉 Softmax 機率 -> 算 Loss -> 往回 Backward -> Adam 更新
    loss = gd(model, optimizer, alien_language, step, num_steps)
    
    if (step + 1) % 10 == 0:
        print(f"訓練第 {step+1:2d} 步 | 誤差 (Loss): {loss:6.4f}")

print("\n=== 訓練完成！我們來考試吧 ===")

# --- 步驟 D：驗證模型學到了什麼 ---
def predict_next(current_word):
    # 把字丟給模型算分數
    logits = model(current_word, pos_id=0, keys=[], values=[])
    # 把分數轉成機率
    probs = softmax(logits) 
    
    print(f"當看到字【{current_word}】時，模型預測下一個字的機率為：")
    for i, p in enumerate(probs):
         # 用簡單的符號標示最高機率
         mark = " ⭐(最高)" if p.data > 0.5 else ""
         print(f"  -> 是【{i}】的機率: {p.data * 100:5.1f}% {mark}")
    print("-" * 30)

# 測試字典裡的三個字
predict_next(0)
predict_next(1)
predict_next(2)