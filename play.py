#!/usr/bin/env python3
"""
Interactive Snake game player.
Play manually with keyboard controls or watch agents play.
"""

import pygame
import argparse
import sys
from env import SnakeEnv
from ui import SnakeRenderer
from agents import RandomAgent, GreedyAgent, AStarAgent
from env.snake_env import Action


def play_human(board_size: int = 10, fps: int = 10):
    """
    Play Snake with keyboard controls.

    Controls:
    - Arrow keys or WASD: Control direction (relative to snake heading)
    - Space: Restart game
    - ESC: Quit

    Args:
        board_size: Size of the game board
        fps: Frames per second
    """
    env = SnakeEnv(board_size=board_size, state_type='feature')
    renderer = SnakeRenderer(board_size=board_size, cell_size=30, fps=fps)

    print("=" * 50)
    print("SNAKESENSE - Human Player Mode")
    print("=" * 50)
    print("\nControls:")
    print("  LEFT ARROW / A : Turn left")
    print("  RIGHT ARROW / D : Turn right")
    print("  UP ARROW / W : Continue straight")
    print("  SPACE : Restart game")
    print("  ESC : Quit")
    print("\nStarting game...")

    running = True
    while running:
        obs = env.reset()
        game_over = False

        while not game_over and running:
            # Get game state for rendering
            state = env.get_state()

            # Render
            renderer.render(
                snake=state['snake'],
                food=state['food'],
                score=state['score'],
                steps=state['steps'],
                direction=state['direction'],
                game_over=False
            )

            # Get user input
            action = None
            waiting = True

            while waiting and running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        waiting = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                            waiting = False
                        elif event.key in [pygame.K_UP, pygame.K_w]:
                            action = Action.STRAIGHT
                            waiting = False
                        elif event.key in [pygame.K_LEFT, pygame.K_a]:
                            action = Action.LEFT
                            waiting = False
                        elif event.key in [pygame.K_RIGHT, pygame.K_d]:
                            action = Action.RIGHT
                            waiting = False

                if waiting:
                    pygame.time.wait(10)

            if action is not None and running:
                # Take step
                obs, reward, done, info = env.step(action)

                if done:
                    game_over = True

                    # Render final state
                    state = env.get_state()
                    renderer.render(
                        snake=state['snake'],
                        food=state['food'],
                        score=state['score'],
                        steps=state['steps'],
                        direction=state['direction'],
                        game_over=True
                    )

                    print(f"\nGame Over!")
                    print(f"Final Score: {info['score']}")
                    print(f"Steps Survived: {info['steps']}")

                    # Wait for restart or quit
                    restart = renderer.wait_for_restart()
                    if not restart:
                        running = False

    renderer.close()
    print("\nThanks for playing SnakeSense!")


def play_agent(agent_type: str, board_size: int = 10, fps: int = 10, num_games: int = 1):
    """
    Watch an agent play Snake.

    Args:
        agent_type: Type of agent ('random', 'greedy', 'astar')
        board_size: Size of the game board
        fps: Frames per second
        num_games: Number of games to play
    """
    env = SnakeEnv(board_size=board_size, state_type='feature')
    renderer = SnakeRenderer(board_size=board_size, cell_size=30, fps=fps)

    # Create agent
    if agent_type == 'random':
        agent = RandomAgent(action_space_size=3, seed=42)
    elif agent_type == 'greedy':
        agent = GreedyAgent(action_space_size=3, env_ref=env)
    elif agent_type == 'astar':
        agent = AStarAgent(action_space_size=3, env_ref=env)
    else:
        print(f"Unknown agent type: {agent_type}")
        sys.exit(1)

    print("=" * 50)
    print(f"SNAKESENSE - {agent_type.upper()} Agent")
    print("=" * 50)
    print(f"\nWatching {agent_type} agent play {num_games} game(s)...")
    print("Press ESC to quit early\n")

    scores = []
    steps_list = []

    for game in range(num_games):
        obs = env.reset()
        agent.reset()
        done = False

        print(f"Game {game + 1}/{num_games} starting...")

        while not done:
            # Check if user wants to quit
            if renderer.check_quit():
                print("\nQuitting early...")
                renderer.close()
                return

            # Get game state for rendering
            state = env.get_state()

            # Render
            renderer.render(
                snake=state['snake'],
                food=state['food'],
                score=state['score'],
                steps=state['steps'],
                direction=state['direction'],
                game_over=False
            )

            # Agent chooses action
            action = agent.act(obs)

            # Take step
            obs, reward, done, info = env.step(action)

        # Game over
        scores.append(info['score'])
        steps_list.append(info['steps'])

        print(f"  Score: {info['score']}, Steps: {info['steps']}")

        # Render final state
        state = env.get_state()
        renderer.render(
            snake=state['snake'],
            food=state['food'],
            score=state['score'],
            steps=state['steps'],
            direction=state['direction'],
            game_over=True
        )

        # Wait a bit before next game
        if game < num_games - 1:
            pygame.time.wait(2000)

    renderer.close()

    # Print statistics
    print("\n" + "=" * 50)
    print("STATISTICS")
    print("=" * 50)
    print(f"Games played: {num_games}")
    print(f"Average score: {sum(scores) / len(scores):.2f}")
    print(f"Max score: {max(scores)}")
    print(f"Average steps: {sum(steps_list) / len(steps_list):.2f}")


def main():
    parser = argparse.ArgumentParser(description="Play Snake interactively")
    parser.add_argument(
        '--mode',
        type=str,
        choices=['human', 'agent'],
        default='human',
        help='Play mode: human or agent'
    )
    parser.add_argument(
        '--agent',
        type=str,
        choices=['random', 'greedy', 'astar'],
        default='greedy',
        help='Agent type (if mode=agent)'
    )
    parser.add_argument(
        '--board-size',
        type=int,
        default=10,
        help='Board size (default: 10)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=10,
        help='Frames per second (default: 10)'
    )
    parser.add_argument(
        '--games',
        type=int,
        default=1,
        help='Number of games to play (agent mode only)'
    )

    args = parser.parse_args()

    if args.mode == 'human':
        play_human(board_size=args.board_size, fps=args.fps)
    else:
        play_agent(
            agent_type=args.agent,
            board_size=args.board_size,
            fps=args.fps,
            num_games=args.games
        )


if __name__ == '__main__':
    main()
