from PIL import Image, ImageDraw
import os

def draw_state(game, debug_text, prefix='', path = r'C:\Users\Stanny\OneDrive - Trifork B.V\Documents\Tag'):

    # define start variables
    screen_size = 400
    grid_size = screen_size / (game._grid_size + 2)

    # size of image
    canvas = (screen_size, screen_size)

    # init canvas
    im = Image.new('RGBA', canvas, (255, 255, 255, 255))
    draw = ImageDraw.Draw(im)

    for l in range(len(game._x_list)):

        x = game._x_list[l]
        y = game._y_list[l]
        t = game._taggers[l]

        if t == 1:
            x_cirle  = grid_size * (x + 1) + l * 3
            y_circle = screen_size - grid_size * (y + 1) + l * 3
            draw.ellipse((x_cirle-5, y_circle-5, x_cirle+5, y_circle+5), fill=(255, 255, 0), outline=(255, 255, 0))
        elif t == 0:
            x_cirle  = grid_size * (x + 1) + l * 3
            y_circle = screen_size - grid_size * (y + 1) + l * 3
            draw.ellipse((x_cirle-5, y_circle-5, x_cirle+5, y_circle+5), fill=(0, 0, 255), outline=(0, 0, 0))

    # Now draw debug text
    draw.text((5,5), debug_text, (0,0,0))

    # save image
    im.save(os.path.join(path,prefix+"{:03d}".format(game._tot_turns)+'.png'))