from tools.base import BaseTool

class PygameGameSetupTool(BaseTool):
    name = "pygame_game_setup"
    description = "A tool to automate the installation of pygame and the creation of a basic game structure for pygame projects."
    parameters = {
        "type": "object",
        "properties": {
            "project_name": {
                "type": "string",
                "description": "The name of the pygame project."
            }
        },
        "required": ["project_name"]
    }

    def run(self, **kwargs) -> str:
        import os
        import subprocess

        project_name = kwargs.get("project_name")
        if not project_name:
            return "Error: 'project_name' is required."

        try:
            # Install pygame
            subprocess.check_call(["pip", "install", "pygame"])

            # Create project directory
            os.makedirs(project_name, exist_ok=True)

            # Create a basic game structure
            main_file_content = f"""import pygame
import sys

# Initialize Pygame
pygame.init()

# Set up the game window
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("{project_name}")

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Fill the screen with black
    screen.fill((0, 0, 0))
    pygame.display.flip()

# Quit Pygame
pygame.quit()
sys.exit()
"""

            with open(os.path.join(project_name, "main.py"), "w") as main_file:
                main_file.write(main_file_content)

            return f"Successfully set up the pygame project '{project_name}'."

        except subprocess.CalledProcessError:
            return "Error: Failed to install pygame. Please check your environment."
        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"

TOOL = PygameGameSetupTool()