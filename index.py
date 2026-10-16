import os
from http.server import BaseHTTPRequestHandler


class handler(BaseHTTPRequestHandler):

  def do_GET(self):
    self.send_response(200)
    self.send_header("Content-type", "text/html; charset=utf-8")
    self.end_headers()
    html_file = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_file):
      with open(html_file, "rb") as f:
        self.wfile.write(f.read())
    else:
      self.wfile.write(
          b"<h1>Socio-Economic Inequality Mining & Analysis</h1>"
      )


def app(environ, start_response):
  html_file = os.path.join(os.path.dirname(__file__), "index.html")
  if os.path.exists(html_file):
    with open(html_file, "rb") as f:
      body = f.read()
  else:
    body = b"<h1>Socio-Economic Inequality Mining & Analysis</h1>"
  start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
  return [body]


application = app
