import pygame
import imageio
import numpy as np
from src.envs.env import SearchAndRescueEnv


def render_episode(env: SearchAndRescueEnv, policy, output_path="outputs/gifs/episode.gif"):
    pygame.init()
    screen = pygame.display.set_mode((400, 400))
    frames = []
    td = env.reset()
    done = False
    while not done:
        screen.fill((255, 255, 255))  # White background
        scale = 400 / env.map_size
        # Draw agents
        for pos in env.agent_pos:
            pygame.draw.circle(screen, (0, 0, 255), (int(pos[0] * scale), int(pos[1] * scale)), 5)
        # Draw victims
        for i, pos in enumerate(env.victim_pos):
            if not env.victims_rescued[i]:
                pygame.draw.circle(screen, (255, 0, 0), (int(pos[0] * scale), int(pos[1] * scale)), 5)
        # Draw obstacles
        for pos in env.obstacles:
            pygame.draw.rect(screen, (0, 0, 0), (int(pos[0] * scale), int(pos[1] * scale), 5, 5))
        # Draw safe zone
        pygame.draw.circle(screen, (0, 255, 0), (int(env.safe_zone[0] * scale), int(env.safe_zone[1] * scale)), 10)
        # Capture frame
        frame = pygame.surfarray.array3d(screen)
        frames.append(np.transpose(frame, (1, 0, 2)))
        td = policy(td)
        td = env.step(td)
        done = td["done"].all().item()
    pygame.quit()
    imageio.mimsave(output_path, frames, fps=30)
