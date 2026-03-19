import random
import math

# --- 使用你提供的爬山演算法主體 ---
def hillClimbing(s, maxGens, maxFails):   
    print("start: ", s.str())             
    fails = 0                             
    for gens in range(maxGens):
        snew = s.neighbor()               
        sheight = s.height()              
        nheight = snew.height()           
        if (nheight >= sheight):          
            print(gens, ':', snew.str())  
            s = snew                      
            fails = 0                     
        else:                             
            fails = fails + 1             
        if (fails >= maxFails):
            break
    print("solution: ", s.str())          
    return s                              

# --- 定義 TSP 的狀態類別 ---
class TSPState:
    def __init__(self, path, cities):
        self.path = path      # 目前的路徑 (例如: [0, 2, 1, 3, 4])
        self.cities = cities  # 城市座標清單

    def distance(self):
        """計算這條路徑的總距離 (包含最後一個城市回到起點)"""
        dist = 0
        n = len(self.path)
        for i in range(n):
            # 取得相鄰兩個城市的座標
            c1 = self.cities[self.path[i]]
            c2 = self.cities[self.path[(i + 1) % n]] # 取餘數確保能從最後一點連回起點
            # 計算兩點間的歐幾里得距離
            dist += math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
        return dist

    def height(self):
        """
        高度函數：因為爬山演算法是找最大值，
        但我們要找最短距離，所以回傳「負的距離」。
        距離越短，負值越大 (爬得越高)。
        """
        return -self.distance()

    def neighbor(self):
        """產生鄰近解：隨機交換路徑中的兩個城市"""
        new_path = list(self.path) # 複製一份目前的路徑
        # 隨機挑選兩個不同的索引位置
        i, j = random.sample(range(len(self.path)), 2)
        # 交換這兩個位置的城市
        new_path[i], new_path[j] = new_path[j], new_path[i]
        
        # 回傳一個新的 TSPState 物件
        return TSPState(new_path, self.cities)

    def str(self):
        """將狀態格式化為字串輸出"""
        return f"距離: {self.distance():.2f}, 路徑: {self.path}"

# --- 主程式執行區塊 ---
if __name__ == "__main__":
    # 1. 隨機生成 10 個城市的 (x, y) 座標
    num_cities = 10
    cities = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(num_cities)]
    
    # 2. 建立初始路徑 (依序走訪 0 到 9)
    initial_path = list(range(num_cities))
    # 可以將初始路徑打亂，增加隨機性
    random.shuffle(initial_path)
    
    # 3. 初始化 TSP 狀態
    initial_state = TSPState(initial_path, cities)
    
    # 4. 呼叫爬山演算法 (設定最多迭代 10000 代，連續失敗 500 次則停止)
    print("=== 開始執行 TSP 爬山演算法 ===")
    hillClimbing(initial_state, maxGens=10000, maxFails=500)