from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name='index'),
    path('getResponse', views.getResponse, name='getResponse'),
    path("chat/", views.chat, name='chat')
]
