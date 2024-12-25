# README

## Squid Games: Beginner Python Implementation

This project contains two simple games inspired by Squid Game, implemented in Python for beginners. The games included
are **Red Light, Green Light** and the **Cookie Game**. This README explains how the games work, how to play, and how to
set them up.

---

### Red Light, Green Light

**How it works:**

- The game alternates between `Green Light` and `Red Light` states.
- During `Green Light`, players can move freely without penalty.
- During `Red Light`, players must stop moving entirely.
- The game tracks how long each player remains still during `Red Light` and calculates their overall time. Moving
  during `Red Light` results in immediate disqualification.
- The player who accumulates the longest time of successful stillness wins.

**How to play:**

1. When the game starts, you will see instructions indicating the current state (`Green Light` or `Red Light`).
2. During `Green Light`, you can move without restriction.
3. As soon as `Red Light` is called, stop moving immediately to avoid disqualification.
4. The game will notify you of your result at the end.

---

### Cookie Game

**How it works:**

- In this game, you are given a virtual cookie with one of four shapes randomly assigned: **Triangle**, **Circle**, *
  *Star**, or **Umbrella**.
- Your objective is to "cut out" the shape without breaking the cookie.
- A computer vision model (YOLO) detects your progress and checks if the shape is intact.
- Breaking the cookie results in a loss, while successfully cutting the shape wins the game.

**How to play:**

1. When the game starts, the program assigns a random shape.
2. Follow the on-screen instructions to simulate cutting out the shape.
3. Be careful not to "break" the cookie (as detected by the computer vision model).
4. The game will notify you if you win or lose.

---

### Setup Instructions

**Prerequisites:**

- Python 3.x installed on your system
- Basic familiarity with running Python scripts

**Steps to run:**

1. Clone the repository:
   ```bash
   git clone https://github.com/MertAkguel/Squid_Game.git
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. The YOLO weights required for the Cookie Game will be downloaded automatically when the game starts. No additional
   setup is needed for this.

4. Run the game scripts:
     ```bash
     python main.py
     ```


5. Follow the on-screen instructions to play the games.

---

### Notes

- These games are simplified for educational purposes and do not represent the full complexity of real-world
  implementations.
- Ensure your environment has internet access for downloading the YOLO weights when running the Cookie Game.

Have fun playing the games!
