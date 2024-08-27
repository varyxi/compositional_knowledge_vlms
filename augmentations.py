import random
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from PIL import Image, ImageOps, ImageEnhance
import cv2

from utils import find_divisors


class AugImage():
    def __init__(self, img_path):
        self.image = Image.open(img_path)

    def change_light(self, img):
        '''
        Randomly changes the sharpness, brightness and contrast
        Input:
            img: image for transformations
            type: PIL Image
        Output:
            transformed image
        '''
        img = ImageEnhance.Brightness(img).enhance(random.choice(np.arange(0.5, 2, 0.5)))
        img = ImageOps.autocontrast(img, cutoff = random.choice(range(-5, 5)), ignore = random.choice(range(-5, 5)))
        img = ImageEnhance.Sharpness(img).enhance(random.choice(range(-2, 5)))
        return img

    def change_size(self, img):
        '''
        Randomly changes size, angle and crop
        Input:
            img: image for transformations
            type: PIL Image
        Output:
            transformed image
        '''
        img = np.array(img)
        image_center = tuple(np.array(img.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, random.choice(np.arange(-3, 3, 0.5)), 1.0)
        result_image = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags = cv2.INTER_LINEAR)
        img = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))

        width, height = img.size
        img = img.resize((int(width*random.choice(np.arange(0, 0.05, 0.01))), int(height*random.choice(np.arange(0, 0.05, 0.01)))))

        left = int(random.choice(np.arange(0, 0.05, 0.01)))*width
        top = int(random.choice(np.arange(0, 0.05, 0.01)))*height
        right = int(1 - random.choice(np.arange(0, 0.05, 0.01)))*width
        bottom = int(1 - random.choice(np.arange(0, 0.05, 0.01)))*height
        img = img.crop((left, top, right, bottom))
        return img

    def mix_parts(self, img):
        '''
        Randomly mixes lines or columns
        Input:
            img: image for transformations
            type: PIL Image
        Output:
            transformed image
        '''
        width, height = img.size
        mix_type = random.choice([0,1])
        array_img = np.array(img)
        if mix_type:
            # mix lines
            divizors = find_divisors(array_img.shape[0])
            if len(divizors) == 0:
                random.shuffle(array_img)
            else:
                array_imgs = np.vsplit(array_img, random.choice(divizors))
                random.shuffle(array_imgs)
                array_img = np.vstack(array_imgs)
        else:
            # mix columns
            divizors = find_divisors(array_img.shape[1])    
            if len(divizors) == 0:
                random.shuffle(array_img)
            else:
                array_imgs = np.hsplit(array_img, random.choice(divizors))
                random.shuffle(array_imgs)
                array_img = np.column_stack(array_imgs)

        return Image.fromarray(np.uint8(np.array(array_img)))

    def aug_image(self):
        '''
        Main function which augments the image
        '''
        change = random.choice([0,1])
        if change:
            return self.mix_parts(self.change_size(self.change_light(self.image)))
        else:
            return self.mix_parts(self.image)


class AugText():
    def __init__(self, text):
        self.words = word_tokenize(text)
        self.alphabet = 'abcdefghijkmnlopqrstuwxyz'

    def noize_word(self, word):
        '''
        Makes mistakes in the word
        Input:
            word: word for noize
            type: str
        Output:
            noisy word
        '''
        mistakes_count = random.choice(range(3))
        if mistakes_count == 0 or len(word) < 3:
            return word
        new_word = ''
        mistakes_places = random.sample(range(len(word)), mistakes_count)
        for i, symb in enumerate(word):
            if i in mistakes_places:
                new_word += random.choice(self.alphabet)
            else:
                new_word += symb
        return new_word

    def mix_words(self, words):
        '''
        Swaps words in random order
        Input:
            words: list of words in sentence
            type: list
        Output:
            shuffled words
        '''
        random.shuffle(words)
        return

    def mix_objects(self, words):
        '''
        Swaps nouns in random order
        Input:
            words: list of words in sentence
            type: list
        Output:
            shuffled words
        '''
        tagged = nltk.pos_tag(words)
        nouns = {i: pair[0] for i, pair in enumerate(tagged) if pair[1]=='NN'}
        if len(nouns) < 2:
            return words
        shuffled_keys = list(nouns.keys())
        random.shuffle(shuffled_keys)
        shuffled = {key: value for key, value in zip(shuffled_keys, list(nouns.values()))}
        new_words = []
        for i in shuffled:
            words[i] = shuffled[i]
        return

    def aug_text(self):
        '''
        Main function which augments the text
        '''
        mix_type = random.choice([0,1])
        if mix_type:
            self.mix_words(self.words)
        else:
            self.mix_objects(self.words)
        for i, word in enumerate(self.words):
            make_noize = random.choice([0]*19+[1])
            if make_noize:
                self.words[i] = self.noize_word(word)
        return ' '.join(self.words)