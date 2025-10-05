# game.py
# 使用 pygame 實作貪食蛇，並與 controls.py 整合（可由 controller.py 在背景修改方向）
# 使用方式：
# 1. 建議在虛擬環境安裝 requirements.txt（包含 pygame）
# 2. 執行：python3 game.py

import pygame
import sys
import random
from controls import up, down, left, right, get_direction, reset as controls_reset

# 視窗設定
WINDOW_SIZE = 400
TILE_COUNT = 20
TILE_SIZE = WINDOW_SIZE // TILE_COUNT

# 顏色
BG = (17, 17, 17)
SNAKE_COLOR = (0, 200, 0)
FOOD_COLOR = (200, 0, 0)
GRID_COLOR = (40, 40, 40)
TEXT_COLOR = (220, 220, 220)

# 遊戲速度（移動間隔 ms）
MOVE_INTERVAL = 120

class SnakeGame:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption('貪食蛇（Python + 控制模組）')
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)
        self.reset()

        # 建立一個 timer event 來定時移動蛇
        self.MOVE_EVENT = pygame.USEREVENT + 1
        pygame.time.set_timer(self.MOVE_EVENT, MOVE_INTERVAL)

    def reset(self):
        self.snake = [ {'x':9, 'y':10}, {'x':8,'y':10}, {'x':7,'y':10} ]
        self.direction = 'right'
        controls_reset()
        self.place_food()
        self.running = True
        self.game_over = False

    def place_food(self):
        while True:
            self.food = { 'x': random.randint(0, TILE_COUNT-1), 'y': random.randint(0, TILE_COUNT-1) }
            if not any(s['x']==self.food['x'] and s['y']==self.food['y'] for s in self.snake):
                break

    def handle_key(self, key):
        # 同步本地鍵盤控制到 controls 模組
        if key in (pygame.K_UP, pygame.K_w):
            up()
        elif key in (pygame.K_DOWN, pygame.K_s):
            down()
        elif key in (pygame.K_LEFT, pygame.K_a):
            left()
        elif key in (pygame.K_RIGHT, pygame.K_d):
            right()
        elif key == pygame.K_r:
            self.reset()

    def move_snake(self):
        # 從 controls 模組取得最新方向
        dirc = get_direction()
        if dirc is None:
            dirc = self.direction
        self.direction = dirc

        mapv = {
            'up': (0, -1),
            'down': (0, 1),
            'left': (-1, 0),
            'right': (1, 0)
        }
        v = mapv.get(self.direction, (1,0))
        head = { 'x': (self.snake[0]['x'] + v[0]) % TILE_COUNT, 'y': (self.snake[0]['y'] + v[1]) % TILE_COUNT }

        # 自撞檢查
        if any(s['x']==head['x'] and s['y']==head['y'] for s in self.snake):
            self.game_over = True
            self.running = False
            return

        self.snake.insert(0, head)
        # 吃到食物
        if head['x'] == self.food['x'] and head['y'] == self.food['y']:
            self.place_food()
        else:
            self.snake.pop()

    def draw(self):
        self.screen.fill(BG)
        # 網格
        for x in range(0, WINDOW_SIZE, TILE_SIZE):
            pygame.draw.line(self.screen, GRID_COLOR, (x,0), (x, WINDOW_SIZE))
        for y in range(0, WINDOW_SIZE, TILE_SIZE):
            pygame.draw.line(self.screen, GRID_COLOR, (0,y), (WINDOW_SIZE, y))

        # 食物
        pygame.draw.rect(self.screen, FOOD_COLOR, (self.food['x']*TILE_SIZE, self.food['y']*TILE_SIZE, TILE_SIZE, TILE_SIZE))

        # 蛇
        for s in self.snake:
            pygame.draw.rect(self.screen, SNAKE_COLOR, (s['x']*TILE_SIZE+1, s['y']*TILE_SIZE+1, TILE_SIZE-2, TILE_SIZE-2))

        # 資訊
        txt = self.font.render(f'長度: {len(self.snake)} 方向: {get_direction()}', True, TEXT_COLOR)
        self.screen.blit(txt, (8,8))

        if self.game_over:
            go = self.font.render('Game Over — 按 R 重開', True, TEXT_COLOR)
            rect = go.get_rect(center=(WINDOW_SIZE//2, WINDOW_SIZE//2))
            self.screen.blit(go, rect)

        pygame.display.flip()

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    self.handle_key(event.key)
                elif event.type == self.MOVE_EVENT and self.running:
                    self.move_snake()

            self.draw()
            self.clock.tick(60)

if __name__ == '__main__':
    game = SnakeGame()
    game.run()
