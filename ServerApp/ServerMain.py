#Server side of image converter application
#Wait for request
#Parse user request
#Do image conversions with NUMPY
#???
#Send results of conversion back to user app (or error messages I guess)
import os
import cherrypy
import numpy as np

import CoreImageConverter.ImageModification as ImMod
from PIL import Image
from jinja2 import Environment, FileSystemLoader


def create_tmp_path (file_name: str)->str:
    return "./images_upload/" + create_tmp_name(file_name)

def create_tmp_name (file_name: str)->str:
    if file_name.endswith('.jpg'):
        return file_name.replace('.jpg', '') + "_tmp" + ".jpg"
    elif file_name.endswith('.png'):
        return file_name.replace('.png', '') + "_tmp" + ".png"
    elif file_name.endswith('.jpeg'):
        return file_name.replace('.jpeg', '') + "_tmp" + ".jpeg"
    else:
        return 'tmp_' + file_name


class MyWebServer(object):
    def __init__(self):
        env_path = os.path.abspath(os.path.join(os.getcwd(), "public", "html_files"))
        files = os.listdir('./images_upload')
        for file in files:
            filepath = os.path.abspath(os.path.join(os.getcwd(), 'images_upload', file))
            print(filepath)
            if os.path.isfile(filepath):
                os.remove(filepath)
        self.jinja_env = Environment(loader=FileSystemLoader(env_path))

    @cherrypy.expose
    def index(self):
        return self.jinja_env.get_template('index.html').render()

    @cherrypy.expose
    def save_upload(self, file):
        save_path = './images_upload/' + file.filename
        with open(save_path, 'wb') as out:
            while True:
                data = file.file.read()
                if not data:
                    break
                out.write(data)
        return self.generate_form(file.filename)

    @cherrypy.expose
    def save_changed_image(self, file: np.ndarray, file_path: str):
        img = Image.fromarray(file)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(file_path)

    @cherrypy.expose
    def generate_form(self, file_name):
        file_path = './images_upload/' + file_name
        template = self.jinja_env.get_template('ImageConvertForm.html')
        file_dims = Image.open(file_path).size
        html_page = template.render(image_path=file_path, width=file_dims[0], height=file_dims[1], file_name=file_name)

        return html_page

    @cherrypy.expose
    def convert_rgb_to_greyscale (self, file_name: str):
        self.save_changed_image(ImMod.convert_rgb_to_greyscale(ImMod.open_image(file_name)), create_tmp_path(file_name))
        return self.jinja_env.get_template('ShowResult.html').render(result_image_path=create_tmp_path(file_name),
                                                                     prev_image_name=file_name,
                                                                     result_image_name=create_tmp_name(file_name))

    @cherrypy.expose
    def crop_image (self, file_name: str, point_a_x: int, point_a_y: int, point_b_x: int, point_b_y: int):
        self.save_changed_image(ImMod.crop_image(ImMod.open_image(file_name), [int(point_a_x), int(point_a_y)],
                                                 [int(point_b_x), int(point_b_y)]), create_tmp_path(file_name))
        return self.jinja_env.get_template('ShowResult.html').render(result_image_path=create_tmp_path(file_name),
                                                                     prev_image_name=file_name,
                                                                     result_image_name=create_tmp_name(file_name))

    @cherrypy.expose
    def resize_image (self, file_name: str, new_width: int, new_height: int):
        self.save_changed_image(ImMod.resize_image(ImMod.open_image(file_name), int(new_height), int(new_width)), create_tmp_path(file_name))
        return self.jinja_env.get_template('ShowResult.html').render(result_image_path=create_tmp_path(file_name),
                                                                     prev_image_name=file_name,
                                                                     result_image_name=create_tmp_name(file_name))

    @cherrypy.expose
    def flip_image (self, file_name: str):
        self.save_changed_image(ImMod.flip_image(ImMod.open_image(file_name)), create_tmp_path(file_name))
        return self.jinja_env.get_template('ShowResult.html').render(result_image_path=create_tmp_path(file_name),
                                                                     prev_image_name = file_name, result_image_name = create_tmp_name(file_name))

    @cherrypy.expose
    def convert_to_negative(self, file_name: str):
        self.save_changed_image(ImMod.convert_to_negative(ImMod.open_image(file_name)), create_tmp_path(file_name))
        return self.jinja_env.get_template('ShowResult.html').render(result_image_path=create_tmp_path(file_name),
                                                                     prev_image_name=file_name,
                                                                     result_image_name=create_tmp_name(file_name))

    @cherrypy.expose
    def solarize_image (self, file_name: str, thresh_val: int = 130, effect: int = 255):
        self.save_changed_image(ImMod.solarize_image(ImMod.open_image(file_name), int(thresh_val), int(effect)), create_tmp_path(file_name))
        return self.jinja_env.get_template('ShowResult.html').render(result_image_path=create_tmp_path(file_name),
                                                                     prev_image_name=file_name,
                                                                     result_image_name=create_tmp_name(file_name))

    @cherrypy.expose
    def change_brightness (self, file_name: str, brightness_value: int = 0):
        self.save_changed_image(ImMod.change_brightness(ImMod.open_image(file_name), int(brightness_value)), create_tmp_path(file_name))
        return self.jinja_env.get_template('ShowResult.html').render(result_image_path=create_tmp_path(file_name),
                                                                     prev_image_name=file_name,
                                                                     result_image_name=create_tmp_name(file_name))

    @cherrypy.expose
    def change_contrast (self, file_name: str, contrast_value: float = 0):
        self.save_changed_image(ImMod.change_contrast(ImMod.open_image(file_name), float(contrast_value)), create_tmp_path(file_name))
        return self.jinja_env.get_template('ShowResult.html').render(result_image_path=create_tmp_path(file_name),
                                                                     prev_image_name=file_name,
                                                                     result_image_name=create_tmp_name(file_name))





if __name__ == "__main__":
    root_dir = os.path.abspath(os.getcwd())
    config = {
        '/': {
            'tools.staticdir.root': root_dir,
        },
        '/images': {
            'tools.staticdir.on': True,
            'tools.staticdir.dir': './images',
        },
        '/images_upload': {
            'tools.staticdir.on': True,
            'tools.staticdir.dir': './images_upload',
        },
        '/logs':{
            'log.access_file': './logs/access',
            'log.error_file': './logs/error',
        },
    }
    cherrypy.tree.mount(MyWebServer(), '/', config)

    cherrypy.engine.start()
    cherrypy.engine.block()
