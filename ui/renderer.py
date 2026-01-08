"""
Pygame renderer for Snake game visualization
"""

import pygame
import numpy as np
from typing import Optional, Tuple


class SnakeRenderer:
    """Pygame-based renderer for Snake environment"""

    # Colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    DARK_GREEN = (0, 150, 0)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    GRAY = (128, 128, 128)
    DARK_GRAY = (64, 64, 64)

    def __init__(self, board_size: int, cell_size: int = 30, fps: int = 10):
        """
        Initialize pygame renderer.

        Args:
            board_size: Size of the game board
            cell_size: Size of each cell in pixels
            fps: Frames per second for rendering
        """
        self.board_size = board_size
        self.cell_size = cell_size
        self.fps = fps

        # Calculate window size
        self.width = board_size * cell_size
        self.height = board_size * cell_size + 60  # Extra space for stats

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("SnakeSense")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.large_font = pygame.font.Font(None, 36)

    def render(
        self,
        snake: list,
        food: Tuple[int, int],
        score: int,
        steps: int,
        direction: int,
        game_over: bool = False
    ):
        """
        Render current game state.

        Args:
            snake: List of (row, col) positions for snake
            food: (row, col) position of food
            score: Current score
            steps: Number of steps taken
            direction: Current direction (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)
            game_over: Whether game is over
        """
        # Clear screen
        self.screen.fill(self.BLACK)

        # Draw grid
        self._draw_grid()

        # Draw snake
        self._draw_snake(snake, direction)

        # Draw food
        self._draw_food(food)

        # Draw stats
        self._draw_stats(score, steps, len(snake))

        # Draw game over message if applicable
        if game_over:
            self._draw_game_over(score)

        # Update display
        pygame.display.flip()
        self.clock.tick(self.fps)

    def _draw_grid(self):
        """Draw grid lines"""
        for i in range(self.board_size + 1):
            # Vertical lines
            pygame.draw.line(
                self.screen,
                self.DARK_GRAY,
                (i * self.cell_size, 0),
                (i * self.cell_size, self.width),
                1
            )
            # Horizontal lines
            pygame.draw.line(
                self.screen,
                self.DARK_GRAY,
                (0, i * self.cell_size),
                (self.width, i * self.cell_size),
                1
            )

    def _draw_snake(self, snake: list, direction: int):
        """Draw snake with head and body"""
        for i, (row, col) in enumerate(snake):
            x = col * self.cell_size
            y = row * self.cell_size

            if i == 0:
                # Draw head
                color = self.DARK_GREEN
                pygame.draw.rect(
                    self.screen,
                    color,
                    (x + 2, y + 2, self.cell_size - 4, self.cell_size - 4),
                    border_radius=8
                )

                # Draw eyes based on direction
                eye_size = 3
                if direction == 0:  # UP
                    eye1_pos = (x + self.cell_size // 3, y + self.cell_size // 3)
                    eye2_pos = (x + 2 * self.cell_size // 3, y + self.cell_size // 3)
                elif direction == 1:  # RIGHT
                    eye1_pos = (x + 2 * self.cell_size // 3, y + self.cell_size // 3)
                    eye2_pos = (x + 2 * self.cell_size // 3, y + 2 * self.cell_size // 3)
                elif direction == 2:  # DOWN
                    eye1_pos = (x + self.cell_size // 3, y + 2 * self.cell_size // 3)
                    eye2_pos = (x + 2 * self.cell_size // 3, y + 2 * self.cell_size // 3)
                else:  # LEFT
                    eye1_pos = (x + self.cell_size // 3, y + self.cell_size // 3)
                    eye2_pos = (x + self.cell_size // 3, y + 2 * self.cell_size // 3)

                pygame.draw.circle(self.screen, self.WHITE, eye1_pos, eye_size)
                pygame.draw.circle(self.screen, self.WHITE, eye2_pos, eye_size)
            else:
                # Draw body
                color = self.GREEN
                pygame.draw.rect(
                    self.screen,
                    color,
                    (x + 3, y + 3, self.cell_size - 6, self.cell_size - 6),
                    border_radius=5
                )

    def _draw_food(self, food: Tuple[int, int]):
        """Draw food"""
        row, col = food
        x = col * self.cell_size + self.cell_size // 2
        y = row * self.cell_size + self.cell_size // 2
        radius = self.cell_size // 3

        pygame.draw.circle(self.screen, self.RED, (x, y), radius)

    def _draw_stats(self, score: int, steps: int, length: int):
        """Draw statistics at the bottom"""
        y_offset = self.width + 10

        # Score
        score_text = self.font.render(f"Score: {score}", True, self.WHITE)
        self.screen.blit(score_text, (10, y_offset))

        # Steps
        steps_text = self.font.render(f"Steps: {steps}", True, self.WHITE)
        self.screen.blit(steps_text, (10, y_offset + 25))

        # Length
        length_text = self.font.render(f"Length: {length}", True, self.WHITE)
        self.screen.blit(length_text, (self.width - 120, y_offset))

    def _draw_game_over(self, score: int):
        """Draw game over message"""
        # Semi-transparent overlay
        overlay = pygame.Surface((self.width, self.height))
        overlay.set_alpha(180)
        overlay.fill(self.BLACK)
        self.screen.blit(overlay, (0, 0))

        # Game over text
        game_over_text = self.large_font.render("GAME OVER", True, self.RED)
        text_rect = game_over_text.get_rect(center=(self.width // 2, self.height // 2 - 40))
        self.screen.blit(game_over_text, text_rect)

        # Final score
        score_text = self.font.render(f"Final Score: {score}", True, self.WHITE)
        score_rect = score_text.get_rect(center=(self.width // 2, self.height // 2 + 10))
        self.screen.blit(score_text, score_rect)

        # Restart instruction
        restart_text = self.font.render("Press SPACE to restart or ESC to quit", True, self.GRAY)
        restart_rect = restart_text.get_rect(center=(self.width // 2, self.height // 2 + 50))
        self.screen.blit(restart_text, restart_rect)

    def close(self):
        """Close pygame window"""
        pygame.quit()

    def check_quit(self) -> bool:
        """
        Check if user wants to quit.

        Returns:
            True if user pressed quit, False otherwise
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return True
        return False

    def wait_for_restart(self) -> bool:
        """
        Wait for user to press space (restart) or ESC (quit).

        Returns:
            True to restart, False to quit
        """
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        return True
                    if event.key == pygame.K_ESCAPE:
                        return False
            self.clock.tick(30)
        return False
