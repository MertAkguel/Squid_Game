from gtts import gTTS
import pygame
import sys
from pygame.locals import *
import random
from ultralytics import YOLO
from helper_functions import *

person_detection_model = YOLO("yolo11n-seg.pt")
MOTION_THRESHOLD = 3000  # Adjust sensitivity as needed

if not os.path.exists("out.mp3"):
    tts = gTTS(text="You moved. You are out.", lang='en')
    tts.save("out.mp3")

# Initialize pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Set up screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Squid Game Start Menu")

# Fonts
font = pygame.font.Font(None, 74)
button_font = pygame.font.Font(None, 50)

# Render "START" button
button_rect = pygame.Rect(SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 - 50, 200, 100)


def draw_button(text, color, y_offset=0):
    button = pygame.Rect(SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 - 50 + y_offset, 200, 100)
    pygame.draw.rect(screen, color, button)
    button_text = button_font.render(text, True, WHITE)
    screen.blit(button_text, (button.x + 50, button.y + 25))
    return button


def toggle_fullscreen():
    global screen, is_fullscreen
    if is_fullscreen:
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        is_fullscreen = False
    else:
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN)
        is_fullscreen = True


def game_flow():
    # Ask how many players
    screen.fill(BLACK)
    title_text = font.render("How Many Players?", True, WHITE)
    screen.blit(title_text, (SCREEN_WIDTH // 2 - title_text.get_width() // 2, 100))
    pygame.display.update()

    # Capture the number of players
    num_players = None
    while num_players is None:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN and event.unicode.isdigit():
                num_players = int(event.unicode)
                break

    # Ask for player names
    players = []
    for i in range(num_players):
        screen.fill(BLACK)
        title_text = font.render(f"Player {i + 1} Name:", True, WHITE)
        screen.blit(title_text, (SCREEN_WIDTH // 2 - title_text.get_width() // 2, 100))
        pygame.display.update()

        player_name = ""
        name_entered = False
        while not name_entered:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        if player_name.strip():
                            print(f"Captured name: {player_name.strip()}")  # Debug: Print entered name
                            players.append(player_name.strip())
                            print(f"Players list: {players}")  # Debug: Print current player list
                            name_entered = True
                    elif event.key == pygame.K_BACKSPACE:
                        player_name = player_name[:-1]
                    else:
                        player_name += event.unicode

            # Update screen with typed name
            screen.fill(BLACK)
            title_text = font.render(f"Player {i + 1} Name:", True, WHITE)
            screen.blit(title_text, (SCREEN_WIDTH // 2 - title_text.get_width() // 2, 100))
            name_text = font.render(player_name, True, GREEN)
            screen.blit(name_text, (SCREEN_WIDTH // 2 - name_text.get_width() // 2, 200))
            pygame.display.update()

    # Play the game for each player
    scores = []
    for player in players:
        screen.fill(BLACK)
        ready_text = font.render(f"{player}, Press Enter to Start", True, WHITE)
        screen.blit(ready_text, (SCREEN_WIDTH // 2 - ready_text.get_width() // 2, SCREEN_HEIGHT // 2 - 50))
        pygame.display.update()

        ready = False
        while not ready:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == KEYDOWN and event.key == pygame.K_RETURN:
                    ready = True

        # Start the game for the player
        score = play_red_light_green_light()  # Update this function to return the player's score
        scores.append((player, score))

        # Save score and check for high score
        save_to_csv(player, score)
        if check_high_score(player, score):
            high_score_text = font.render("New High Score!", True, GREEN)
            screen.blit(high_score_text,
                        (SCREEN_WIDTH // 2 - high_score_text.get_width() // 2, SCREEN_HEIGHT // 2 + 50))
            pygame.display.update()
            pygame.time.wait(2000)

    # Determine the winner
    winner = max(scores, key=lambda x: x[1])
    screen.fill(BLACK)
    winner_text = font.render(f"{winner[0]} Wins with {winner[1]}s!", True, GREEN)
    screen.blit(winner_text, (SCREEN_WIDTH // 2 - winner_text.get_width() // 2, SCREEN_HEIGHT // 2 - 50))
    pygame.display.update()
    pygame.time.wait(1500)

    # Back to Main Menu
    main_menu()


# Main menu function
def main_menu():
    global is_fullscreen
    is_fullscreen = True

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN:
                if event.key == pygame.K_SPACE:
                    toggle_fullscreen()
            if event.type == MOUSEBUTTONDOWN:
                if button_rect.collidepoint(event.pos):
                    game_flow()

        screen.fill(BLACK)
        title_text = font.render("Squid Game", True, WHITE)
        screen.blit(title_text, (SCREEN_WIDTH // 2 - title_text.get_width() // 2, 100))
        draw_button("START", RED)
        pygame.display.update()


# Placeholder for the game action
# Red Light, Green Light Game Logic
def play_red_light_green_light():
    # Display initial game instructions
    screen.fill(BLACK)
    instructions = [
        "Red Light: Stop moving!",
        "Green Light: Move forward!",
    ]
    for i, line in enumerate(instructions):
        line_text = font.render(line, True, WHITE)
        screen.blit(line_text, (SCREEN_WIDTH // 2 - line_text.get_width() // 2, 150 + i * 50))
    pygame.display.update()
    pygame.time.wait(1500)

    # Open the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Start game simulation
    red_light = False
    prev_masked_frame = None
    still_in_game = True
    game_timer_start = None
    grace_period = True  # Grace period for the first 10 seconds
    grace_period_start = pygame.time.get_ticks()  # Track the start time of grace period
    timer_font = pygame.font.Font(None, 40)  # Font for the game timer

    while still_in_game:

        green_duration = random.randint(2, 5)
        red_duration = random.randint(2, 5)

        # Alternate between Green Light and Red Light
        if not red_light:
            duration = green_duration
            mask_color = GREEN
            color = GREEN
            status = "You can Move!"

        else:
            duration = red_duration
            mask_color = (0, 0, 255)
            color = (255, 0, 0)
            status = "Do not Move!"

        # Record start time for this phase
        start_time = pygame.time.get_ticks()

        while (pygame.time.get_ticks() - start_time) // 1000 < duration:

            # Capture frame from camera
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            results = person_detection_model.predict(frame, classes=[0])

            for result in results:
                for mask in result.masks.xy:
                    points = np.int32([mask])

                    binary_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.polylines(frame, [points], isClosed=True, color=mask_color, thickness=1)
                    cv2.fillPoly(binary_mask, [points], WHITE)  # Fill the mask with white (255)

                    if prev_masked_frame is not None:
                        # Calculate motion magnitude
                        motion_magnitude = calculate_motion(prev_masked_frame, frame, binary_mask)

                        # Check if motion exceeds the threshold
                        if motion_magnitude > MOTION_THRESHOLD and red_light and not grace_period:
                            os.system("out.mp3")
                            still_in_game = False
                        else:
                            pass

            # Update grace period status
            elapsed_grace = (pygame.time.get_ticks() - grace_period_start) // 1000
            if elapsed_grace >= 10 and grace_period:
                grace_period = False
                game_timer_start = pygame.time.get_ticks()

            # Timer display
            if not grace_period:
                elapsed_game_time = (pygame.time.get_ticks() - game_timer_start) // 1000
                timer_text = timer_font.render(f"Time: {elapsed_game_time}s", True, WHITE)
            else:
                timer_text = timer_font.render(f"Grace Period: {10 - elapsed_grace}s", True, WHITE)

            # Update the previous masked frame
            prev_masked_frame = frame.copy()

            # Convert OpenCV frame (BGR) to Pygame Surface (RGB)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_surface = pygame.image.frombuffer(frame.tobytes(), frame.shape[1::-1], "RGB")

            # Display status on Pygame screen
            screen.fill(BLACK)

            # Draw the camera feed on the left
            camera_rect = pygame.Rect(0, 0, SCREEN_WIDTH // 2, SCREEN_HEIGHT)
            screen.blit(pygame.transform.scale(frame_surface, (camera_rect.width, camera_rect.height)), camera_rect)

            # Draw game status on the right
            game_rect = pygame.Rect(SCREEN_WIDTH // 2, 0, SCREEN_WIDTH // 2, SCREEN_HEIGHT)
            pygame.draw.rect(screen, BLACK, game_rect)

            status_text = font.render(status, True, color)
            screen.blit(status_text, (SCREEN_WIDTH // 2 - status_text.get_width() // 2, SCREEN_HEIGHT // 2 - 50))
            screen.blit(timer_text, (SCREEN_WIDTH // 2 + 20, 20))  # Top-right corner for timer

            pygame.display.update()

            # Check for quit event in OpenCV window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        red_light = not red_light

    if not still_in_game:
        elapsed_game_time = (pygame.time.get_ticks() - game_timer_start) // 1000
        return elapsed_game_time


# Placeholder for Game Two
def game_two():
    screen.fill(BLACK)
    action_text = font.render("Game Two Coming Soon...", True, WHITE)
    screen.blit(action_text, (SCREEN_WIDTH // 2 - action_text.get_width() // 2, SCREEN_HEIGHT // 2 - 50))
    pygame.display.update()
    pygame.time.wait(2000)
    main_menu()


if __name__ == "__main__":
    # Run the menu
    main_menu()
