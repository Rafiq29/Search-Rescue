import pygame
import imageio
import numpy as np
from tensordict import TensorDict
import torch
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def render_episode(env, policy, output_path="outputs/gifs/episode.gif", max_steps=100):
    """Render an episode and save as GIF"""
    pygame.init()
    screen_size = 400
    screen = pygame.display.set_mode((screen_size, screen_size))
    pygame.display.set_caption("Search and Rescue")

    frames = []
    td = env.reset()
    done = False
    step_count = 0

    print(f"Starting episode rendering with {env.num_agents} agents and {env.num_victims} victims")

    while not done and step_count < max_steps:
        # Clear screen
        screen.fill((255, 255, 255))  # White background

        # Calculate scaling factor
        scale = screen_size / env.map_size

        # Draw grid
        for i in range(env.map_size + 1):
            pygame.draw.line(screen, (200, 200, 200),
                             (i * scale, 0), (i * scale, screen_size), 1)
            pygame.draw.line(screen, (200, 200, 200),
                             (0, i * scale), (screen_size, i * scale), 1)

        # Draw obstacles
        for x in range(env.map_size):
            for y in range(env.map_size):
                if env.grid[x, y] == 1:  # Obstacle
                    rect = pygame.Rect(x * scale, y * scale, scale, scale)
                    pygame.draw.rect(screen, (50, 50, 50), rect)

        # Draw safe zones
        if hasattr(env, 'safe_zones') and len(env.safe_zones) > 0:
            for pos in env.safe_zones:
                center = (int((pos[0] + 0.5) * scale), int((pos[1] + 0.5) * scale))
                pygame.draw.circle(screen, (0, 255, 0), center, int(scale * 0.4))
                pygame.draw.circle(screen, (0, 200, 0), center, int(scale * 0.4), 2)

        # Draw victims (only unrescued ones)
        if hasattr(env, 'victim_pos') and hasattr(env, 'victims_rescued'):
            for i, pos in enumerate(env.victim_pos):
                if not env.victims_rescued[i]:
                    center = (int((pos[0] + 0.5) * scale), int((pos[1] + 0.5) * scale))
                    pygame.draw.circle(screen, (255, 0, 0), center, int(scale * 0.3))
                    pygame.draw.circle(screen, (200, 0, 0), center, int(scale * 0.3), 2)

        # Draw agents
        if hasattr(env, 'agent_pos'):
            colors = [(0, 0, 255), (0, 100, 255), (100, 0, 255), (0, 200, 255), (200, 0, 255)]
            for i, pos in enumerate(env.agent_pos):
                color = colors[i % len(colors)]
                center = (int((pos[0] + 0.5) * scale), int((pos[1] + 0.5) * scale))
                pygame.draw.circle(screen, color, center, int(scale * 0.25))
                pygame.draw.circle(screen, (0, 0, 0), center, int(scale * 0.25), 2)

                # Draw agent number
                font = pygame.font.Font(None, int(scale * 0.3))
                text = font.render(str(i), True, (255, 255, 255))
                text_rect = text.get_rect(center=center)
                screen.blit(text, text_rect)

        # Add text info
        font = pygame.font.Font(None, 24)
        info_text = f"Step: {step_count}, Rescued: {sum(env.victims_rescued)}/{len(env.victims_rescued)}"
        text_surface = font.render(info_text, True, (0, 0, 0))
        screen.blit(text_surface, (10, 10))

        pygame.display.flip()

        # Capture frame
        frame = pygame.surfarray.array3d(screen)
        frames.append(np.transpose(frame, (1, 0, 2)))

        # Get action from policy
        try:
            td = policy(td)
            td = env.step(td)
            done = td["done"].all().item() if td["done"].numel() > 0 else False
            step_count += 1
        except Exception as e:
            print(f"Error during step {step_count}: {e}")
            break

    pygame.quit()

    # Save as GIF
    if frames:
        print(f"Saving {len(frames)} frames to {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        imageio.mimsave(output_path, frames, fps=5, duration=0.2)
        print(f"GIF saved successfully to {output_path}")
    else:
        print("No frames captured!")


def render_episode_simple(env, policy, output_path="outputs/gifs/episode.gif", max_steps=50):
    """Simple text-based rendering for debugging"""
    frames = []
    td = env.reset()
    done = False
    step_count = 0

    print(f"Starting simple episode with {env.num_agents} agents")

    while not done and step_count < max_steps:
        print(f"\nStep {step_count}:")
        print(f"Agent positions: {env.agent_pos}")
        print(f"Victim positions: {env.victim_pos}")
        print(f"Victims rescued: {env.victims_rescued}")

        # Get action from policy
        td = policy(td)
        td = env.step(td)
        done = td["done"].all().item() if td["done"].numel() > 0 else False
        step_count += 1

        if done:
            print(f"Episode finished at step {step_count}")
            break

    return step_count