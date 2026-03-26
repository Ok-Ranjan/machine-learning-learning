import gymnasium as gym
import flappy_bird_gymnasium
import pygame

# Creating env
env = gym.make("FlappyBird-v0", render_mode="human")
state, _ = env.reset()
done = False

# Initializing Pygame Kyboard
pygame.init()
screen = pygame.display.get_surface()  # Gym has already created a window

while not done:
    action = 0 # default -> 0 is not flap & 1 -> is flap

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                action = 1 # flap
    
    state, reward, done, terminated, info = env.step(action)
    env.render()

env.close()
pygame.quit()