import pygame
import os

pygame.mixer.init()

# Inisialisasi suara hanya sekali
BEEP_PATH = os.path.join("utils", "beep.mp3")
pygame.mixer.music.load(BEEP_PATH)

def bip():
    # Pastikan suara tidak tumpang tindih
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.play()
