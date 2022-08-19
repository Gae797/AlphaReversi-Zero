'''
This class creates and updates the window to handle the GUI
'''

import tkinter as tk
from PIL import Image, ImageDraw, ImageTk

from src.environment.board import Board
from src.environment.gui.ui import UI, SIDE

class Window:

    def __init__(self, draw_legal_moves=False):

        self.ui = UI(draw_legal_moves)

        self.root = tk.Tk()
        self.root.title("Board")

        self.canvas = tk.Canvas(self.root,width=SIDE,height=SIDE)
        self.canvas.pack()

    def update(self, board):

        self.ui.update_pieces(board)

        image = ImageTk.PhotoImage(self.ui.image)
        self.canvas.delete("all")
        sprite = self.canvas.create_image(SIDE//2,SIDE//2,image=image)

        self.root.update()

    def close(self):

        self.root.quit()
        self.root.destroy()
