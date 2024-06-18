# blog/routing.py

from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/chatgpt-demo/$', consumers.ChatConsumer.as_asgi()),
    # Adjust the path and consumer class as per your setup
]
