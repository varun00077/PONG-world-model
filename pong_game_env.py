# Custom 32x32 Pong environment used to collect data for the world model
import numpy as np


class My_pong:

    # ---------------- GAME SETTINGS ----------------

    SIZE = 32
    MAX_STEPS = 120

    NUM_ACTIONS = 3
    LEFT = 0
    NOTHING = 1
    RIGHT = 2

    # Paddle
    PADDLE_HEIGHT = 2
    PADDLE_WIDTH = 8
    PADDLE_Y = 29
    PADDLE_V = 2

    # Ball
    BALL_RADIUS = 1.5
    MIN_SPEED = 0.35
    MAX_SPEED = 0.95

    # RGB colors
    BACKGROUND = (0.05, 0.05, 0.08)
    WALL_COLOR = (0.35, 0.35, 0.40)
    PADDLE_COLOR = (0.20, 0.85, 0.80)
    BALL_COLOR = (1.00, 0.85, 0.30)

    # ---------------- INITIALIZATION ----------------

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

        self.y_grid, self.x_grid = np.mgrid[
            0:self.SIZE, 0:self.SIZE
        ]

        self.reset()

    # ---------------- RESET GAME ----------------

    def reset(self):

        # Random initial paddle position
        self.px = float(
            self.rng.integers(
                6, self.SIZE - 6 - self.PADDLE_WIDTH
            )
        )

        # Random initial ball position
        self.ball_x = float(self.rng.uniform(6, self.SIZE - 6))
        self.ball_y = float(self.rng.uniform(4, 12))

        # Random initial velocity
        speed_x = self.rng.uniform(self.MIN_SPEED, self.MAX_SPEED)
        speed_y = self.rng.uniform(self.MIN_SPEED, self.MAX_SPEED)

        direction = self.rng.choice([-1, 1])

        self.ball_vx = float(direction * speed_x)
        self.ball_vy = float(speed_y)

        self.steps = 0
        self.last_hit = False

        return self.observe()

    # ---------------- RENDER OBSERVATION ----------------

    def observe(self):

        # Create an RGB image: height, width, channels
        image = np.empty(
            (self.SIZE, self.SIZE, 3),
            dtype=np.float32
        )

        image[...] = self.BACKGROUND

        # Top wall
        image[0, :] = self.WALL_COLOR

        # Left wall
        image[:, 0] = self.WALL_COLOR

        # Right wall
        image[:, -1] = self.WALL_COLOR

        # Paddle
        paddle_x = int(round(self.px))

        image[
            self.PADDLE_Y:self.PADDLE_Y + self.PADDLE_HEIGHT,
            paddle_x:paddle_x + self.PADDLE_WIDTH
        ] = self.PADDLE_COLOR

        # Squared distance from each pixel to the ball center
        distance_sq = (
            (self.x_grid - self.ball_x) ** 2
            + (self.y_grid - self.ball_y) ** 2
        )

        # Gaussian glow
        glow = 1.5 * np.exp(
            -distance_sq / (2 * self.BALL_RADIUS ** 2)
        )

        # Add the ball's RGB color
        ball_color = np.asarray(
            self.BALL_COLOR, dtype=np.float32
        )

        image += glow[..., None] * ball_color

        return np.clip(image, 0.0, 1.0).astype(np.float32)

    # ---------------- MOVE PADDLE ----------------

    def move_paddle(self, action):

        if action == self.LEFT:
            self.px -= self.PADDLE_V

        elif action == self.RIGHT:
            self.px += self.PADDLE_V

        elif action == self.NOTHING:
            pass

        else:
            raise ValueError("Action must be 0, 1, or 2.")

        # Keep the paddle inside the side boundaries
        min_x = 1
        max_x = self.SIZE - 1 - self.PADDLE_WIDTH

        self.px = float(np.clip(self.px, min_x, max_x))

    # ---------------- MOVE BALL ----------------

    def move_ball(self):

        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

    # ---------------- WALL COLLISIONS ----------------

    def handle_side_and_top_walls(self):

        radius = self.BALL_RADIUS

        # Left wall
        if self.ball_x < radius:
            self.ball_x = 2 * radius - self.ball_x
            self.ball_vx = abs(self.ball_vx)

        # Right wall
        elif self.ball_x > self.SIZE - 1 - radius:
            right_limit = self.SIZE - 1 - radius
            self.ball_x = 2 * right_limit - self.ball_x
            self.ball_vx = -abs(self.ball_vx)

        # Top wall
        if self.ball_y < radius:
            self.ball_y = 2 * radius - self.ball_y
            self.ball_vy = abs(self.ball_vy)

    def handle_bottom_wall(self):

        # Keep the ball bouncing for a continuous game.
        bottom_limit = self.SIZE - 1 - self.BALL_RADIUS

        if self.ball_y > bottom_limit:
            self.ball_y = 2 * bottom_limit - self.ball_y
            self.ball_vy = -abs(self.ball_vy)

    # ---------------- PADDLE COLLISION ----------------

    def handle_paddle_collision(self, action):

        radius = self.BALL_RADIUS

        # Ball must be moving downward and reach paddle height
        reaches_paddle = (
            self.ball_vy > 0
            and self.ball_y + radius >= self.PADDLE_Y
        )

        # Ball's horizontal edges
        ball_left = self.ball_x - radius
        ball_right = self.ball_x + radius

        # Paddle's horizontal edges
        paddle_left = self.px
        paddle_right = self.px + self.PADDLE_WIDTH

        # Check whether their horizontal ranges overlap
        overlaps = (
            ball_right >= paddle_left
            and ball_left <= paddle_right
        )

        if reaches_paddle and overlaps:

            # Reflect the ball across the contact boundary
            contact_y = self.PADDLE_Y - radius
            self.ball_y = 2 * contact_y - self.ball_y

            # Send ball upward
            self.ball_vy = -abs(self.ball_vy)

            # Optional: let the action influence horizontal direction
            if action == self.LEFT:
                self.ball_vx = -abs(self.ball_vx)

            elif action == self.RIGHT:
                self.ball_vx = abs(self.ball_vx)

            return True

        return False

    # ---------------- ONE TIME STEP ----------------

    def step(self, action):

        # Move paddle based on chosen action
        self.move_paddle(action)

        # Advance ball position
        self.move_ball()

        # Handle side and top wall collisions first
        self.handle_side_and_top_walls()

        # Paddle collision before bottom-wall collision
        hit_paddle = self.handle_paddle_collision(action)

        # If the ball did not hit the paddle, the bottom wall
        # still makes it bounce to keep the game continuous.
        if not hit_paddle:
            self.handle_bottom_wall()

        # Remember whether the paddle hit the ball (useful as a reward later)
        self.last_hit = hit_paddle

        # Count time steps
        self.steps += 1

        done = self.steps >= self.MAX_STEPS

        return self.observe(), done
