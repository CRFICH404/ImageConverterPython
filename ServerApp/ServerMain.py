#Server side of image converter application
#Wait for request
#Parse user request
#Do image conversions with NUMPY
#???
#Send results of conversion back to user app (or error messages I guess)
import os
from functools import singledispatch

import cherrypy
import numpy as np
import CoreImageConverter.ImageModification as immod
from PIL import Image
from jinja2 import Environment, FileSystemLoader

from CoreImageConverter.ImageModification import resize_image


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
    def save_image(self, file, file_name = None):
        if file_name is None:
            file_path = './images_upload/' + file.filename
        else:
            file_path = './images_upload/' + file_name
            img = Image.fromarray(file)
            img.save(file_path)
            return
        size = 0
        with open(file_path, 'wb') as out:
            while True:
                data = file.file.read(8192)
                if not data:
                    break
                out.write(data)
                size += len(data)

    @cherrypy.expose
    def save_image_and_change (self, file):
        print(file.filename)
        self.save_image(file)
        return self.generate_form(file.filename)

    @cherrypy.expose
    def generate_form(self, file_name):
        file_path = './images_upload/' + file_name
        template = self.jinja_env.get_template('ImageConvertForm.html')
        file_dims = Image.open(file_path).size
        html_page = template.render(image_path=file_path, width=file_dims[0], height=file_dims[1], image_name=file_name)

        return html_page

    @cherrypy.expose
    def convert_image(self, **kwargs):
        result_image = None
        try:
            image_name = kwargs['image_name']
            command = kwargs['command']
            result_image = immod.change_image(**kwargs)
            result_image_name = image_name
            if image_name.endswith('.jpg'):
                result_image_name = image_name.replace('.jpg', '') + "_tmp" + ".jpg"
            elif image_name.endswith('.png'):
                result_image_name = image_name.replace('.png', '') + "_tmp" + ".png"
            elif image_name.endswith('.jpeg'):
                result_image_name = image_name.replace('.jpeg', '') + "_tmp" + ".jpeg"
            if result_image_name != image_name:
                self.save_image(result_image, result_image_name)
            template = self.jinja_env.get_template('ShowResult.html')
            result_image_path = "./images_upload/" + result_image_name
            html_page = template.render(result_image_path=result_image_path)
            return html_page

        except KeyError as e:
            return {f"Missing parameter: {e}"}


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
