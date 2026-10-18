from django.urls import path
from Chatbot import views

urlpatterns = [
    path('', views.index,name='users-index'),
    path('chatbot/', views.chatbot_response_view,name='chatbot'),
]


# from django.urls import path
# from . import views

# urlpatterns = [
#     path('chatbot/', views.chatbot_response_view, name='chatbot'),
# ]