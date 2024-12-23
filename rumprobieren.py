from gtts import gTTS
import os
import pygame
import sys
from pygame.locals import *
import cv2
import numpy as np
import random
from ultralytics import YOLO

person_detection_model = YOLO("yolo11n-seg.pt")
MOTION_THRESHOLD = 3000  # Adjust sensitivity as needed

if not os.path.exists("out.mp3"):
    tts = gTTS(text="You are out", lang='en')
    tts.save("out.mp3")

# Initialize pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (0, 0, 255)
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


# Main menu function
def main_menu():
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == MOUSEBUTTONDOWN:
                if button_rect.collidepoint(event.pos):
                    red_light_green_light()

        # Background
        screen.fill(BLACK)

        # Title text
        title_text = font.render("Squid Game", True, WHITE)
        screen.blit(title_text, (SCREEN_WIDTH // 2 - title_text.get_width() // 2, 100))

        # Draw start button
        draw_button("START", RED)

        pygame.display.update()


# Red Light, Green Light game introduction
def red_light_green_light():
    play_button = pygame.Rect(SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 - 50, 200, 100)
    skip_button = pygame.Rect(SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 + 100, 200, 100)

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == MOUSEBUTTONDOWN:
                if play_button.collidepoint(event.pos):
                    play_red_light_green_light()
                elif skip_button.collidepoint(event.pos):
                    game_two()

        # Background
        screen.fill(BLACK)

        # Title text
        title_text = font.render("Red Light, Green Light", True, WHITE)
        screen.blit(title_text, (SCREEN_WIDTH // 2 - title_text.get_width() // 2, 100))

        # Draw Play and Skip buttons
        draw_button("PLAY", GREEN)
        draw_button("SKIP", RED, 150)

        pygame.display.update()


def calculate_motion(prev_frame, curr_frame, mask):
    # Apply the mask to both frames
    prev_roi = cv2.bitwise_and(prev_frame, prev_frame, mask=mask)
    curr_roi = cv2.bitwise_and(curr_frame, curr_frame, mask=mask)

    # Convert to grayscale for simplicity
    prev_gray = cv2.cvtColor(prev_roi, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_roi, cv2.COLOR_BGR2GRAY)

    # Calculate absolute difference
    diff = cv2.absdiff(prev_gray, curr_gray)

    # Threshold the difference to binarize
    _, diff_thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Calculate the motion magnitude (sum of non-zero pixels)
    motion_magnitude = np.sum(diff_thresh > 0)
    return motion_magnitude


# Placeholder for the game action
# Red Light, Green Light Game Logic
def play_red_light_green_light():
    # Display initial game instructions
    screen.fill(BLACK)
    instructions = [
        "Red Light: Stop moving!",
        "Green Light: Move forward!",
        "Wait for further instructions on-screen."
    ]
    for i, line in enumerate(instructions):
        line_text = font.render(line, True, WHITE)
        screen.blit(line_text, (SCREEN_WIDTH // 2 - line_text.get_width() // 2, 150 + i * 50))
    pygame.display.update()
    pygame.time.wait(5000)

    # Open the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Start game simulation
    red_light = False
    prev_masked_frame = None
    still_in_game = True
    while still_in_game:

        green_duration = random.randint(1, 4)
        red_duration = random.randint(2, 6)

        # Alternate between Green Light and Red Light
        if not red_light:
            duration = green_duration
            color = GREEN
        else:
            duration = red_duration
            color = RED

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
                    cv2.polylines(frame, [points], isClosed=True, color=color, thickness=1)

                    binary_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(binary_mask, [points], 255)  # Fill the mask with white (255)

                    if prev_masked_frame is not None:
                        # Calculate motion magnitude
                        motion_magnitude = calculate_motion(prev_masked_frame, frame, binary_mask)

                        # Check if motion exceeds the threshold
                        if motion_magnitude > MOTION_THRESHOLD and red_light:
                            status = "Moving - OUT"
                            color = RED
                            os.system("out.mp3")
                            still_in_game = False
                        else:
                            status = "Still - IN"
                            color = GREEN

                        # Display the status
                        cv2.putText(frame, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Show camera feed in a window
            cv2.imshow("Camera Feed - Press Q to Quit", frame)

            # Update the previous masked frame
            prev_masked_frame = frame.copy()

            # Game logic: Alternate between Red and Green Light
            elapsed_time = (pygame.time.get_ticks() - start_time) // 1000
            if elapsed_time % 5 < 3:
                red_light = False
                status = "GREEN LIGHT - Move!"
            else:
                red_light = True
                status = "RED LIGHT - Stop!"

            # Display status on Pygame screen
            screen.fill(BLACK)
            status_text = font.render(status, True, color)
            screen.blit(status_text, (SCREEN_WIDTH // 2 - status_text.get_width() // 2, SCREEN_HEIGHT // 2 - 50))
            pygame.display.update()

            # Check for quit event in OpenCV window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


# Placeholder for Game Two
def game_two():
    screen.fill(BLACK)
    action_text = font.render("Game Two Coming Soon...", True, WHITE)
    screen.blit(action_text, (SCREEN_WIDTH // 2 - action_text.get_width() // 2, SCREEN_HEIGHT // 2 - 50))
    pygame.display.update()
    pygame.time.wait(2000)
    main_menu()


# Run the menu
main_menu()
