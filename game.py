# Import packages
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import glob
import os

# Helpers
def mold_to_size(value, size):
    return str(value).strip().replace(" ", "")[:size].ljust(size)

# Game
class Game:

    def __init__(self, config, display_path="defaultname"):

        # Game variables
        self._x_list = [-1] * config.number_of_players
        self._y_list = [-1] * config.number_of_players
        self._taggers = ([True] * config.number_of_taggers +
                         [False] * (config.number_of_players - config.number_of_taggers))
        self._options = [0, 1, 2, 3,  # 0:up, 1:down, 2:left, 3:right,
                         4, 5, 6, 7,  # 4:up left, 5:up right, 6:down left, 7:down right
                         8]           # 8:dont move
        self._ended = 0
        self._tag_happened = False

        # Initialize render
        self._rendered = ''
        self._previously_rendered = ''

        # Initialize the save path for displaying games
        self._display_path = display_path

        # Initialize config
        self._config = config

        # Initialize game
        self.init_random_game()

    def init_random_game(self):
        """
        Start a randomized game
        """

        self._x_list = [-1] * self._config.number_of_players
        self._y_list = [-1] * self._config.number_of_players

        occupied = set()
        for i in range(self._config.number_of_players):
            while True:
                x = int(np.floor(random.random() * self._config.grid_size))
                y = int(np.floor(random.random() * self._config.grid_size))
                if (x, y) not in occupied:
                    self._x_list[i] = x
                    self._y_list[i] = y
                    occupied.add((x, y))
                    break

        if random.random() < 0.5:
            self._taggers = [not t for t in self._taggers]
        self._ended = 0
        self._tag_happened = False

    # Move options
    def what_options(self, turn):

        """
        Check what move options the player has
        """

        # Get x and y position of the player whose turn it is
        x = self._x_list[turn]
        y = self._y_list[turn]

        # For different scenario's, rule out options
        options = np.ones(np.shape(self._options), dtype=bool)
        if y == 0:
            options[0] = False
            options[4] = False
            options[5] = False
        if y == self._config.grid_size - 1:
            options[1] = False
            options[6] = False
            options[7] = False
        if x == 0:
            options[2] = False
            options[4] = False
            options[6] = False
        if x == self._config.grid_size - 1:
            options[3] = False
            options[5] = False
            options[7] = False

        return options

    @staticmethod
    def change_position(choice, x, y):

        """
        Change the position of a player on the board
        """

        deltas = {
            0: (0, -1),  # up
            1: (0, 1),  # down
            2: (-1, 0),  # left
            3: (1, 0),  # right
            4: (-1, -1),  # up left
            5: (1, -1),  # up right
            6: (-1, 1),  # down left
            7: (1, 1),  # down right
            8: (0, 0),  # don't move
        }
        dx, dy = deltas.get(choice, (0, 0))
        return x + dx, y + dy

    def move(self, turn, choice):

        """
        Move player on the board and update game params
        """

        # Validate move choice
        valid = [int(i) for i in np.where(self.what_options(turn))[0]]
        if choice not in valid:
            raise ValueError(f"Invalid move choice {choice} for turn {turn}. Valid: {valid}")

        x = self._x_list[turn]
        y = self._y_list[turn]
        x, y = self.change_position(choice, x, y)

        self._y_list[turn] = y
        self._x_list[turn] = x

        reward = self.what_reward(turn, choice)
        return reward

    def what_reward(self, turn, choice):

        """
        Define what reward is associated with the choice of the player
        """

        # Get state of player whose turn it is
        is_tagger = self._taggers[turn]
        x = self._x_list[turn]
        y = self._y_list[turn]

        # Check if player is in the same spot as another player
        in_same_spot = [i for i in range(self._config.number_of_players) if
                        (self._x_list[i] == x) and (self._y_list[i] == y) and (i != turn)]

        # A tagger gets some punishment for each move, a runner gets some reward for each move
        if is_tagger:
            reward = -1 * self._config.step_points
        else:
            reward = self._config.step_points

        # Each player gets some punishment for moving
        if choice in [0, 1, 2, 3]:              # 0:up, 1:down, 2:left, 3:right,
            reward -= self._config.step_points * 0.25
        elif choice in [4, 5, 6, 7]:            # 4:up left, 5:up right, 6:down left, 7:down right
            reward -= ((self._config.step_points * 0.25) ** 2 * 2) ** 0.5
        if choice == 8:                         # 8:dont move
            reward -= 0

        # When the tagger caught the runner, the tagger gets a large reward and the runner a large punishment
        self._tag_happened = False
        for caught in in_same_spot:
            caught_is_tagger = self._taggers[caught]
            if is_tagger != caught_is_tagger:
                if is_tagger:
                    reward += self._config.tag_points
                else:
                    reward -= self._config.tag_points
                self._ended += 1
                self._tag_happened = True

        return round(reward, 2)

    def render(self, prediction=None, state=None, display=False, size=5):

        """
        Render a game for debugging purposes
        """

        # Initialize field
        playing_field = np.full(shape=(self._config.grid_size, self._config.grid_size), fill_value='     ')
        taggers = np.where(self._taggers)[0].tolist()
        runners = np.where([t == False for t in self._taggers])[0].tolist()
        if prediction is not None:
            print_prediction = []
            for p in prediction:
                int_digits = len(str(int(p)))
                if int_digits > size:
                    p = f'{str(int(p))[0]}e{int_digits-1}'
                print_prediction += [mold_to_size(p, size)]

        for tagger in taggers:
            x_tagger = self._x_list[tagger]
            y_tagger = self._y_list[tagger]
            playing_field[x_tagger, y_tagger] = (f"x    {playing_field[x_tagger, y_tagger]}"
                                                 ).strip().replace(" ", "").ljust(size)

            if (prediction is not None) and (state is not None):
                if tagger == state[4]:
                    for i, c in enumerate(print_prediction):
                        x, y = self.change_position(i, x_tagger, y_tagger)
                        if (0 <= x < self._config.grid_size) and (0 <= y < self._config.grid_size):
                            playing_field[x, y] = (playing_field[x, y] + c).strip().replace(" ", "").ljust(size)

        for runner in runners:
            x_runner = self._x_list[runner]
            y_runner = self._y_list[runner]
            if 'x' in playing_field[x_runner, y_runner]:
                playing_field[x_runner, y_runner] = playing_field[x_runner, y_runner].replace('x', '%', 1)
            else:
                playing_field[x_runner, y_runner] = ('o    ' + playing_field[x_runner, y_runner]).strip().replace(" ", "").ljust(size)

            if (prediction is not None) and (state is not None):
                if runner == state[4]:
                    for i, c in enumerate(print_prediction):
                        x, y = self.change_position(i, x_runner, y_runner)
                        if (0 <= x < self._config.grid_size) and (0 <= y < self._config.grid_size):
                            playing_field[x, y] = (playing_field[x, y] + c).strip().replace(" ", "").ljust(size)

        self._rendered = playing_field.T

        # Display the rendered field:
        if display:
            print(self._rendered)

    @staticmethod
    def extract_position(grid, symbol):

        """
        Extract player position
        """

        return [[x,y] for x in range(len(grid)) for y in range(len(grid[x])) if symbol in str(grid[x][y])]

    @staticmethod
    def draw_cat(draw, players, previous_players, color,
                 cell_size, player_radius, ear_ratio=0.8):
        """
        Draw a simple cat to represent the tagger
        """
        top_margin = 0
        # Horizontal ellipse head (wider than tall)
        face_rx = max(3, int(player_radius * 1.6))  # horizontal radius
        face_ry = max(2, int(player_radius * 1.0))  # vertical radius

        # Ear geometry
        ear_size = max(2, int(max(face_rx, face_ry) * ear_ratio))
        ear_base_offset_x = max(1, int(face_rx * 0.30))  # closer to center
        apex_tilt = max(1, int(ear_size * 0.4))

        # Whiskers
        whisker_len = max(4, int(face_rx * 0.9))
        whisker_spread = max(2, int(face_ry * 0.35))
        whisker_width = 2
        cheek_offset_x = max(2, int(face_rx * 0.55))  # where whiskers attach on cheeks
        cheek_y_offset = max(0, int(face_ry * 0.05))  # slight offset from vertical center

        for (x, y), _ in zip(players, previous_players):
            # Draw only at the current position (no trail)
            center_x = x * cell_size + cell_size // 2
            center_y = y * cell_size + top_margin + cell_size // 2

            # Head (horizontal ellipse)
            draw.ellipse(
                [
                    center_x - face_rx,
                    center_y - face_ry,
                    center_x + face_rx,
                    center_y + face_ry,
                ],
                fill=color,
            )

            # Ears base line lowered to overlap into the head
            base_y = center_y - face_ry + max(1, int(face_ry * 0.28))

            # Left ear (apex top-left)
            left_base_cx = center_x - ear_base_offset_x
            draw.polygon(
                [
                    (left_base_cx - ear_size // 2, base_y),         # left base
                    (left_base_cx + ear_size // 2, base_y),         # right base
                    (left_base_cx - apex_tilt, base_y - ear_size),  # apex tilted to top-left
                ],
                fill=color,
            )

            # Right ear (apex top-right)
            right_base_cx = center_x + ear_base_offset_x
            draw.polygon(
                [
                    (right_base_cx - ear_size // 2, base_y),
                    (right_base_cx + ear_size // 2, base_y),
                    (right_base_cx + apex_tilt, base_y - ear_size),  # apex tilted to top-right
                ],
                fill=color,
            )

            # Whiskers from cheeks (left and right)
            cheek_y = center_y + cheek_y_offset
            # Left cheek whiskers (extend to the left)
            draw.line(
                [(center_x - cheek_offset_x, cheek_y),
                 (center_x - cheek_offset_x - whisker_len, cheek_y - whisker_spread)],
                fill="grey", width=whisker_width,
            )
            draw.line(
                [(center_x - cheek_offset_x, cheek_y),
                 (center_x - cheek_offset_x - whisker_len, cheek_y)],
                fill="grey", width=whisker_width,
            )
            draw.line(
                [(center_x - cheek_offset_x, cheek_y),
                 (center_x - cheek_offset_x - whisker_len, cheek_y + whisker_spread)],
                fill="grey", width=whisker_width,
            )
            # Right cheek whiskers (extend to the right)
            draw.line(
                [(center_x + cheek_offset_x, cheek_y),
                 (center_x + cheek_offset_x + whisker_len, cheek_y - whisker_spread)],
                fill="grey", width=whisker_width,
            )
            draw.line(
                [(center_x + cheek_offset_x, cheek_y),
                 (center_x + cheek_offset_x + whisker_len, cheek_y)],
                fill="grey", width=whisker_width,
            )
            draw.line(
                [(center_x + cheek_offset_x, cheek_y),
                 (center_x + cheek_offset_x + whisker_len, cheek_y + whisker_spread)],
                fill="grey", width=whisker_width,
            )

        return draw

    @staticmethod
    def draw_mouse(draw, players, previous_players, color,
                   cell_size, half_size, width=10):
        """
        Draw a simple mouse to represent the runner
        """
        top_margin = 0
        # Scale parameters from half_size
        body_len = max(6, int(half_size * 3.0))         # body length (horizontal)
        body_height = max(4, int(half_size * 1.6))      # body height (vertical)
        nose_radius = max(2, int(body_height * 0.1))   # round nose tip
        ear_radius = max(2, int(body_height * 0.28))    # ears
        tail_len = max(8, int(half_size * 3.2))         # tail length
        tail_width = max(1, width // 3)
        whisker_len = max(6, int(half_size * 1.2))
        whisker_spread = max(2, int(body_height * 0.25))  # vertical spread between whiskers

        # Colors for details
        detail_color = "black"

        for (x, y), (x_previous, y_previous) in zip(players, previous_players):
            center_x = (x_previous + (x - x_previous)) * cell_size + cell_size // 2
            center_y = ((y_previous + (y - y_previous)) * cell_size) + top_margin + cell_size // 2

            # Body (wide ellipse centered slightly behind the nose)
            body_left = center_x - body_len // 2
            body_right = center_x + body_len // 2
            body_top = center_y - body_height // 2
            body_bottom = center_y + body_height // 2
            draw.ellipse([body_left, body_top, body_right, body_bottom], fill=color)

            # Nose (round cap at the front)
            nose_cx = body_right  # rightmost front
            nose_cy = center_y
            draw.ellipse(
                [
                    nose_cx - nose_radius,
                    nose_cy - nose_radius,
                    nose_cx + nose_radius,
                    nose_cy + nose_radius,
                ],
                fill=detail_color,
            )

            # Ears (two small circles near the top-front of the body)
            ear_base_x = center_x + int(body_len * 0.20)
            ear_base_y = center_y - int(body_height * 0.45)
            # Left ear
            draw.ellipse(
                [
                    ear_base_x - ear_radius - ear_radius // 2,
                    ear_base_y - ear_radius,
                    ear_base_x - ear_radius // 2,
                    ear_base_y + ear_radius,
                ],
                fill=color,
            )
            # Right ear
            draw.ellipse(
                [
                    ear_base_x + ear_radius // 2,
                    ear_base_y - ear_radius,
                    ear_base_x + ear_radius + ear_radius // 2,
                    ear_base_y + ear_radius,
                ],
                fill=color,
            )

            # Whiskers (three lines per side from the nose)
            # Left side
            draw.line(
                [(nose_cx, nose_cy), (nose_cx + whisker_len, nose_cy - whisker_spread)],
                fill=detail_color,
                width=max(1, tail_width - 1),
            )
            draw.line(
                [(nose_cx, nose_cy), (nose_cx + whisker_len, nose_cy)],
                fill=detail_color,
                width=max(1, tail_width - 1),
            )
            draw.line(
                [(nose_cx, nose_cy), (nose_cx + whisker_len, nose_cy + whisker_spread)],
                fill=detail_color,
                width=max(1, tail_width - 1),
            )
            # Right side (optional for symmetry; comment out if you prefer one-sided whiskers)
            draw.line(
                [(nose_cx, nose_cy), (nose_cx - whisker_len, nose_cy - whisker_spread)],
                fill=detail_color,
                width=max(1, tail_width - 1),
            )
            draw.line(
                [(nose_cx, nose_cy), (nose_cx - whisker_len, nose_cy)],
                fill=detail_color,
                width=max(1, tail_width - 1),
            )
            draw.line(
                [(nose_cx, nose_cy), (nose_cx - whisker_len, nose_cy + whisker_spread)],
                fill=detail_color,
                width=max(1, tail_width - 1),
            )

            # Curvy tail: start at back center, draw a gentle sine-like polyline
            tail_start_x = body_left
            tail_start_y = center_y
            segments = 12
            amp = max(2, int(body_height * 0.35))  # amplitude of the curve
            tail_pts = []
            for s in range(segments + 1):
                t = s / segments
                # Ease-out to taper curvature near the end
                x = tail_start_x - int(t * tail_len)
                y = tail_start_y + int(amp * 0.5 * np.sin(2 * np.pi * (t + 0.15)))
                tail_pts.append((x, y))
            # Draw the polyline
            for a, b in zip(tail_pts, tail_pts[1:]):
                draw.line([a, b], fill=color, width=tail_width)

        return draw

    def save(self, prefix, text=None):

        """
        Save image snapshot of each step in the game to later record them into a gif
        """

        # Get player (previous) positions
        x_players = self.extract_position(self._rendered, 'x') + self.extract_position(self._rendered, '%')
        o_players = self.extract_position(self._rendered, 'o') + self.extract_position(self._rendered, '%')

        if str(self._previously_rendered) == '':
            self._previously_rendered = self._rendered

        x_previous_players = self.extract_position(self._previously_rendered, 'x') + self.extract_position(self._previously_rendered, '%')
        o_previous_players = self.extract_position(self._previously_rendered, 'o') + self.extract_position(self._previously_rendered, '%')

        # Set cell size and create an empty image
        cell_size = 200
        grid_width = len(self._rendered)
        grid_height = len(self._rendered)
        image_width = grid_width * cell_size
        image_height = grid_height * cell_size
        image = Image.new("RGB", (image_width, image_height), "white")
        draw = ImageDraw.Draw(image)

        # Draw text
        if text is not None:
            font = ImageFont.load_default()
            text_margin = 5
            draw.text((text_margin, text_margin), text, fill=(0, 0, 0), font=font)

        # Draw grid lines
        top_margin = 0
        grid_line_width = 2  # make lines a tiny bit wider
        for i in range(0, image_width, cell_size):
            draw.line([(i, top_margin), (i, image_height)], fill="#c9c9c9", width=grid_line_width)
        for j in range(top_margin, image_height, cell_size):
            draw.line([(0, j), (image_width, j)], fill="#c9c9c9", width=grid_line_width)

        # Draw players
        player_radius = 50
        draw = self.draw_cat(draw, x_players, x_previous_players, "black",
                                         cell_size, player_radius)
        draw = self.draw_mouse(draw, o_players, o_previous_players, "grey",
                                        cell_size, player_radius)

        # Save stationary image
        prefix = str(prefix).zfill(3)
        image.save(os.path.join(self._display_path, f"{prefix}.png"))

        # Save current state as previous
        self._previously_rendered = self._rendered

    def record(self, game_path):

        """
        Record a gif from all saved image snapshots of each step in the game
        """

        try:
            frame_files = sorted(glob.glob(os.path.join(self._display_path, "*.png")))
            if not frame_files:
                print("No frames to record.")
                return

            imgs = [Image.open(fn) for fn in frame_files]
            # duration in ms per frame; adjust as needed instead of duplicating frames
            duration = 150
            gif_save_path = os.path.join(self._display_path, f"{game_path}.gif")
            os.makedirs(os.path.dirname(gif_save_path), exist_ok=True)
            imgs[0].save(
                os.path.join(self._display_path, f"{game_path}.gif"),
                save_all=True,
                optimize=False,
                append_images=imgs[1:],
                loop=0,
                duration=duration,
            )
        except Exception as e:
            print(f"Failed saving to gif: {e}")
        finally:
            # Ensure images are closed before deletion
            for im in locals().get('imgs', []):
                try:
                    im.close()
                except Exception():
                    pass
            try:
                for filename in glob.glob(os.path.join(self._display_path, "*.png")):
                    os.remove(filename)
            except Exception as e:
                print(f"Failed to clean up frames: {e}")

    @property
    def x_list(self):
        return self._x_list

    @property
    def y_list(self):
        return self._y_list

    @property
    def taggers(self):
        return self._taggers

    @property
    def ended(self):
        return self._ended

    @ended.setter
    def ended(self, value):
        self._ended = value

    @property
    def tag_happened(self):
        return self._tag_happened

    @property
    def rendered(self):
        return self._rendered