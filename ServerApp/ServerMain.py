#Server side of image converter application
#Wait for request
#Parse user request
#Do image conversions with NUMPY
#???
#Send results of conversion back to user app (or error messages I guess)
import os
import random
import string
from distutils.command.config import config
from operator import index

import numpy as np
import cherrypy

class MyWebServer(object):
    @cherrypy.expose
    def index(self):
        return open('public/html_files/HomePage.html')

class ImageUploader(object):
    @cherrypy.expose
    def index(self):
        return open('C:\\Users\\zazon\\PycharmProjects\\ImageConverter\\ServerApp\\public\\html_files\\ImageUploadPage.html')
class ImageConvertForm(object):
    def __init__(self):
        self.file = ImageUploader().index



if __name__ == "__main__":
    root_dir = os.path.abspath(os.getcwd())
    config = {
        '/': {
            'tools.staticdir.root': root_dir,
        },
        '/static': {
            'tools.staticdir.on': True,
            'tools.staticdir.dir': './public',
        },
        '/images': {
            'tools.staticdir.on': True,
            'tools.staticdir.dir': './images',
        },
        '/logs':{
            'log.access_file': './logs/access',
            'log.error_file': './logs/error',
        },
    }
    image_upload_config = {
        '/static': {
            'tools.staticdir.on': True,
            'tools.staticdir.dir': root_dir + '/public',
        },
        '/images': {
            'tools.staticdir.on': True,
            'tools.staticdir.dir': root_dir + '/images',
        },
        '/logs':{
            'log.access_file': root_dir + '/logs/access',
            'log.error_file': root_dir + '/logs/error',
        },
    }
    image_convert_config = {
        '/static': {
            'tools.staticdir.on': True,
            'tools.staticdir.dir': root_dir + '/public',
        },
        '/images': {
            'tools.staticdir.on': True,
            'tools.staticdir.dir': root_dir + '/images',
        },
        '/logs': {
            'log.access_file': root_dir + '/logs/access',
            'log.error_file': root_dir + '/logs/error',
        },
    }
    cherrypy.tree.mount(MyWebServer(), '/', config)
    cherrypy.tree.mount(ImageUploader(), '/image_upload', image_upload_config)
    cherrypy.tree.mount(ImageConvertForm(), '/image_convert_form', image_convert_config)

    cherrypy.engine.start()
    cherrypy.engine.block()