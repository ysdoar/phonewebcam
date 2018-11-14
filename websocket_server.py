# -*- coding: utf-8 -*-
import argparse
import random
import os
import datetime

import numpy as np
import cv2
import time
import requests
import threading
from threading import Thread, Event, ThreadError

import cherrypy

from ws4py.server.cherrypyserver import WebSocketPlugin, WebSocketTool
from ws4py.websocket import WebSocket
from ws4py.messaging import TextMessage


class ChatWebSocketHandler(WebSocket):
  def received_message(self, m):
    cherrypy.engine.publish('websocket-broadcast', m)

  def closed(self, code, reason="A client left the room without a proper explanation."):
    cherrypy.engine.publish('websocket-broadcast', TextMessage(reason))


class Root(object):
  def __init__(self, host, port, ssl=False):
    self.host = host
    self.port = port
    self.scheme = 'wss' if ssl else 'ws'

  @cherrypy.expose
  def index(self):
    return """<html>
    <head>
     </head>
    </html>
    """ % {'username': "User%d" % random.randint(0, 100), 'host': self.host, 'port': self.port, 'scheme': self.scheme}

  @cherrypy.expose
  def ws(self):
    cherrypy.log("Handler created: %s" % repr(cherrypy.request.ws_handler))


class Cam():
  def __init__(self, url):
    self.stream = requests.get(url, stream=True)
    self.thread_cancelled = False
    self.thread = Thread(target=self.run)
    print "camera initialised"

  def start(self):
    self.thread.start()
    print "camera stream started"

  def run(self):
    bytes = ''
    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    #recognizer = cv2.createLBPHFaceRecognizer()
    # recognizer.load('trainer/trainer.yml')
    txt_welcome_count = 0
    offset = 50
    img_num = 0
    time_save = False
    now = datetime.datetime.now()

    while not self.thread_cancelled:
      try:
        bytes += self.stream.raw.read(1024)
        a = bytes.find('\xff\xd8')
        b = bytes.find('\xff\xd9')
        if a != -1 and b != -1:
          jpg = bytes[a:b + 2]
          bytes = bytes[b + 2:]
          img = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
          if img == None:
            #raise Exception("could not load image !")
            continue
          # start face recon
          gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
          faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100),
                                               flags=cv2.CASCADE_SCALE_IMAGE)
          for (x, y, w, h) in faces:
            if (not time_save) or (int(round(time.time() - time_save)) > 10):
              time_save = time.time()
              img_num = img_num + 1
              cv2.imwrite("faces/face-" + str(now.hour) + '_' + str(img_num) + ".jpg",
                          gray[y - offset:y + h + offset, x - offset:x + w + offset])
              cherrypy.engine.publish('websocket-broadcast', "faces/face-" + str(now.hour) + '_' + str(img_num) + ".jpg")
            #nbr_predicted, conf = recognizer.predict(gray[y:y + h, x:x + w])
            cv2.rectangle(img, (x - 50, y - 50), (x + w + 50, y + h + 50), (225, 0, 0), 2)
            txt_welcome_count += 1
            # if (nbr_predicted == 1):
            #    nbr_predicted = 'Yaakov'
            # elif (nbr_predicted == 2):
            #    nbr_predicted = 'Anirban'
            # else:
            #    nbr_predicted = 'you'
          if cv2.waitKey(1) == 27:
            exit(0)
      except ThreadError:
        self.thread_cancelled = True

  def is_running(self):
    return self.thread.isAlive()

  def shut_down(self):
    self.thread_cancelled = True
    # block while waiting for thread to terminate
    while self.thread.isAlive():
      time.sleep(1)
    return True


if __name__ == '__main__':
  import logging
  from ws4py import configure_logger
  configure_logger(level=logging.DEBUG)

  parser = argparse.ArgumentParser(description='Echo CherryPy Server')
  parser.add_argument('--host', default='LOCALIP')
  parser.add_argument('-p', '--port', default=91, type=int)
  parser.add_argument('--ssl', action='store_true')
  args = parser.parse_args()

  url_cam = 'http://LOCALIP:8080/video'
  cam = Cam(url_cam)
  cam.start()

  cherrypy.config.update({'server.socket_host': args.host,
                          'server.socket_port': args.port,
                          'tools.staticdir.root': os.path.abspath(os.path.join(os.path.dirname(__file__), 'static'))})

  if args.ssl:
    cherrypy.config.update({'server.ssl_certificate': './server.crt',
                            'server.ssl_private_key': './server.key'})

  WebSocketPlugin(cherrypy.engine).subscribe()
  cherrypy.tools.websocket = WebSocketTool()

  cherrypy.quickstart(Root(args.host, args.port, args.ssl), '', config={
      '/ws': {
          'tools.websocket.on': True,
          'tools.websocket.handler_cls': ChatWebSocketHandler
      },
      '/js': {
          'tools.staticdir.on': True,
          'tools.staticdir.dir': 'js'
      }
  }
  )
