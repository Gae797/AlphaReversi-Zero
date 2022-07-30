from PIL import Image, ImageDraw, ImageTk

from src.environment.board import Board

from src.environment.config import *

SQUARE_UNITS = 80
PIECE_RADIUS = SQUARE_UNITS//2
SIDE = SQUARE_UNITS * BOARD_SIZE

SQUARE_COLOR = (0,150,0)
WHITE_COLOR = (255,255,255)
BLACK_COLOR = (0,0,0)
TRANSPARENT_GREY_COLOR = (125,125,125,100)

class UI:

    def __init__(self, draw_legal_moves=False):

        self.draw_legal_moves = draw_legal_moves

        self.image = Image.new('RGB', (SIDE, SIDE), (128, 128, 128))
        self.board = ImageDraw.Draw(self.image)
        self.pieces = ImageDraw.Draw(self.image)
        self.legal_moves = ImageDraw.Draw(self.image, "RGBA")

        self.draw_board()

    def draw_board(self):

        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                self.board.rectangle((j*SQUARE_UNITS, i*SQUARE_UNITS, (j+1)*SQUARE_UNITS, (i+1)*SQUARE_UNITS),
                                    fill=SQUARE_COLOR,
                                    outline=BLACK_COLOR)

    def update_pieces(self, board):

        self.draw_board()

        white_pieces, black_pieces, turn, legal_moves, reward = board.get_state(legal_moves_format="matrix")

        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):

                if white_pieces[i][j]==1:
                    self.pieces.ellipse((j*SQUARE_UNITS+10, i*SQUARE_UNITS+10, (j+1)*SQUARE_UNITS-10, (i+1)*SQUARE_UNITS-10),
                                        fill=WHITE_COLOR,
                                        outline=BLACK_COLOR,
                                        width=2)

                elif black_pieces[i][j]==1:
                    self.pieces.ellipse((j*SQUARE_UNITS+10, i*SQUARE_UNITS+10, (j+1)*SQUARE_UNITS-10, (i+1)*SQUARE_UNITS-10),
                                        fill=BLACK_COLOR,
                                        outline=WHITE_COLOR)

                if self.draw_legal_moves:
                    if legal_moves[i][j]==1:
                        self.legal_moves.ellipse((j*SQUARE_UNITS+10, i*SQUARE_UNITS+10, (j+1)*SQUARE_UNITS-10, (i+1)*SQUARE_UNITS-10),
                                            fill=TRANSPARENT_GREY_COLOR,
                                            outline=(255,0,0,125))


    def save_image(self):

        self.image.save('test.jpg', quality=100)
