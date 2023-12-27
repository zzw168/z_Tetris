import time

import pygame
import random

# 初始化
pygame.init()

# 设置屏幕尺寸
screen_width = 300
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Tetris Game")

# 定义颜色
black = (0, 0, 0)
white = (255, 255, 255)
red = (255, 0, 0)
blue = (0, 0, 255)
green = (0, 255, 0)
yellow = (255, 255, 0)
orange = (255, 165, 0)
purple = (128, 0, 128)
cyan = (0, 255, 255)

# 定义方块的形状
shapes = [
    [[1, 1, 1, 1]],
    [[1, 1], [1, 1]],
    [[1, 1, 1], [0, 1, 0]],
    [[1, 1, 1], [1, 0, 0]],
    [[1, 1, 1], [0, 0, 1]],
    [[1, 1, 0], [0, 1, 1]],
    [[0, 1, 1], [1, 1, 0]]
]

# 定义方块的颜色
colors = [red, blue, green, yellow, orange, purple, cyan]

# 定义方块的大小
block_size = 30

# 定义游戏区域
play_width = 10
play_height = 20

# 定义游戏区域的起始位置
play_x = (screen_width - play_width * block_size) // 2
play_y = screen_height - play_height * block_size

# 初始化得分
score = 0

# 定义字体和文字大小
font = pygame.font.SysFont(None, 30)


# 定义方块类
# 定义方块类
# 定义方块类
class Block:
    def __init__(self, shape, color):
        self.shape = shape
        self.color = color
        self.rotation = 0
        self.x = play_width // 2 - len(shape[0]) // 2
        self.y = 0

    def rotate(self):
        self.rotation = (self.rotation + 1) % len(self.shape)
        self.shape = self.rotate_matrix(self.shape)

    def rotate_back(self):
        self.rotation = (self.rotation - 1) % len(self.shape)
        self.shape = self.rotate_matrix(self.shape)

    def rotate_matrix(self, matrix):
        return [list(row)[::-1] for row in zip(*matrix)]

    def move_left(self):
        self.x -= 1

    def move_right(self):
        self.x += 1

    def move_down(self):
        self.y += 1

    def draw(self):
        for i in range(len(self.shape[0])):
            for j in range(len(self.shape)):
                if self.shape[j][i] == 1:
                    pygame.draw.rect(screen, self.color, (play_x + (self.x + i) * block_size,
                                                          play_y + (self.y + j) * block_size,
                                                          block_size, block_size))


# 创建一个新的方块
def new_block():
    shape = random.choice(shapes)
    color = random.choice(colors)
    return Block(shape, color)


# 检查方块是否与游戏区域或其他方块重叠
def collision(block, play_area):
    for i in range(len(block.shape[0])):
        for j in range(len(block.shape)):
            if block.shape[j][i] == 1:
                if (block.x + i) < 0 or (block.x + i) >= play_width or \
                        (block.y + j) >= play_height or play_area[block.y + j][block.x + i] != black:
                    return True
    return False


# 绘制游戏区域
def draw_play_area(play_area):
    for i in range(len(play_area)):
        for j in range(len(play_area[i])):
            pygame.draw.rect(screen, play_area[i][j], (play_x + j * block_size,
                                                       play_y + i * block_size,
                                                       block_size, block_size), 0)


# 主游戏循环
def main():
    global score
    clock = pygame.time.Clock()
    game_over = False
    play_area = [[black] * play_width for _ in range(play_height)]
    current_block = new_block()

    # 添加方块下移的计时器和延迟时间
    fall_time = 0
    fall_speed = 0.5  # 方块下移的速度，单位是方块每秒的格数

    # 添加按键状态变量
    key_down_pressed = False

    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    current_block.move_left()
                    if collision(current_block, play_area):
                        current_block.move_right()
                elif event.key == pygame.K_RIGHT:
                    current_block.move_right()
                    if collision(current_block, play_area):
                        current_block.move_left()
                elif event.key == pygame.K_DOWN:
                    key_down_pressed = True
                elif event.key == pygame.K_UP:
                    current_block.rotate()
                    if collision(current_block, play_area):
                        current_block.rotate_back()
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_DOWN:
                    key_down_pressed = False

        # 控制方块下移的速度
        fall_time += clock.get_rawtime()
        clock.tick()

        if key_down_pressed or (fall_time / 1000.0 > fall_speed):
            fall_time = 0

            # 移动方块
            current_block.move_down()
            time.sleep(0.05)
            if collision(current_block, play_area):
                current_block.y -= 1
                for i in range(len(current_block.shape[0])):
                    for j in range(len(current_block.shape)):
                        if current_block.shape[j][i] == 1:
                            play_area[current_block.y + j][current_block.x + i] = current_block.color
                current_block = new_block()
                if collision(current_block, play_area):
                    game_over = True

        # 检查是否有完整的行
        full_lines = []
        for i in range(play_height):
            if all(col != black for col in play_area[i]):
                full_lines.append(i)
        for line in full_lines:
            del play_area[line]
            play_area.insert(0, [black] * play_width)
            score += 1

        # 绘制背景和方块
        screen.fill(black)
        draw_play_area(play_area)
        current_block.draw()

        # 绘制得分
        score_text = font.render(f"Score: {score}", True, white)
        screen.blit(score_text, (10, 10))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
