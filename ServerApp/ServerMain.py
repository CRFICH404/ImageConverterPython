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
        tmp_file_url = "data:image/jpeg, image/jpg, image/png, base64," + base64.b64decode(file.read()).decode('ascii')
        return open("./public/html_files/ImageConvertForm.html")

class MyImage(object):
    @cherrypy.expose
    def index(self, file):
        return f"""
        <img src = "data:image/png;base64,{file}" />
        """

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
    cherrypy.tree.mount(MyWebServer(), '/', config)
    cherrypy.tree.mount(ImageUploader(), '/image_upload', config)

    cherrypy.engine.start()
    cherrypy.engine.block()