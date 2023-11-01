import pygame, sys

BLACK = (0, 0, 0)
WHITE = (200, 200, 200)
GREEN =( 60,179,113)
WINDOW_HEIGHT = 400
WINDOW_WIDTH = 400


def main(self,grid_size):
    global SCREEN, CLOCK
    pygame.init()
    pygame.event.pump()
    SCREEN = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    CLOCK = pygame.time.Clock()
    SCREEN.fill(GREEN)

    running = True

    while True:

        for event in pygame.event.get():

            if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_q:  # Check if the "q" key is pressed
                                running = False
                                pygame.quit()
                                sys.exit()

        drawGrid(self,grid_size)
        pygame.event.pump()
        pygame.display.update()
        CLOCK.tick(30)


def centering(image, new_width, new_height, blockSize, block_x, block_y):
    image=pygame.transform.scale(image, (new_width, new_height))
# Calculate the position (x, y) within the specified grid block
    # Calculate the position to center the image within the grid block
    image_x = block_x + (blockSize - new_width) // 2
    image_y = block_y + (blockSize - new_height) // 2
    return image, (image_x, image_y)

def drawGrid(self,grid_size):
    blockSize = int(400/grid_size) #Set the size of the grid block
    for x in range(0, WINDOW_WIDTH, blockSize):
        for y in range(0, WINDOW_HEIGHT, blockSize):
            rect = pygame.Rect(x, y, blockSize, blockSize)
            pygame.draw.rect(SCREEN, WHITE, rect, 1)
    
    new_width = WINDOW_WIDTH/grid_size
    new_height = WINDOW_WIDTH/grid_size

    hunter = pygame.image.load('hunter.png')
    block_x = self.current_pos[1] * blockSize
    block_y = self.current_pos[0] * blockSize
    hunter, (image_x, image_y) =centering(hunter, new_width, new_height, blockSize, block_x, block_y)
    SCREEN.blit( hunter, (image_x, image_y))

    deer = pygame.image.load('deer1.png')
    block_x = self.hard_goal[1] * blockSize
    block_y = self.hard_goal[0] * blockSize
    deer, (image_x, image_y) =centering(deer, new_width, new_height, blockSize, block_x, block_y)
    SCREEN.blit( deer, (image_x, image_y))


    bunny = pygame.image.load('bunny.png')
    block_x = self.easy_goal[1] * blockSize
    block_y =self.easy_goal[0] * blockSize
    bunny, (image_x, image_y) =centering(bunny, new_width, new_height, blockSize, block_x, block_y)
    SCREEN.blit( bunny, (image_x, image_y))


    pygame.display.flip()

def render_update(self, grid_size):
    blockSize = int(400/grid_size)
    new_width = WINDOW_WIDTH/grid_size
    new_height = WINDOW_WIDTH/grid_size

    hunter = pygame.image.load('hunter.png')
    block_x = self.state[1] * blockSize
    block_y = self.state[0] * blockSize
    hunter, (image_x, image_y) =centering(hunter, new_width, new_height, blockSize, block_x, block_y)
    SCREEN.blit( hunter, (image_x, image_y))

    deer = pygame.image.load('deer1.png')
    block_x = self.hard_target[1] * blockSize
    block_y = self.hard_target[0] * blockSize
    deer, (image_x, image_y) =centering(deer, new_width, new_height, blockSize, block_x, block_y)
    SCREEN.blit( deer, (image_x, image_y))


    bunny = pygame.image.load('bunny.png')
    block_x = self.easy_target[1] * blockSize
    block_y = self.easy_target[0] * blockSize
    bunny, (image_x, image_y) =centering(bunny, new_width, new_height, blockSize, block_x, block_y)
    SCREEN.blit( bunny, (image_x, image_y))
    pygame.display.flip()

    pygame.display.flip()