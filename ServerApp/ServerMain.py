#Server side of image converter application
#Wait for request
#Parse user request
#Do image conversions with NUMPY
#???
#Send results of conversion back to user app (or error messages I guess)
import base64
import os
import numpy
import cherrypy

class MyWebServer(object):
    @cherrypy.expose
    def index(self):
        return open('./public/html_files/HomePage.html')

class ImageUploader(object):
    @cherrypy.expose
    def index(self):
        return open('./public/html_files/ImageUploadPage.html')

    @cherrypy.expose
    def generate_form(self, file):
        file_path ="images_upload/" + file.filename
        size = 0
        with open(file_path, 'wb') as out:
            while True:
                data = file.file.read(8192)
                if not data:
                    break
                out.write(data)
                size += len(data)
        return open("./public/html_files/ImageConvertForm.html")

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
    cherrypy.tree.mount(ImageUploader(), '/image_upload', config)

    cherrypy.engine.start()
    cherrypy.engine.block()

    print(root_dir)
