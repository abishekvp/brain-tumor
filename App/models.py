from django.db import models
from django.contrib.auth.models import User
from PIL import Image


import os


class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    avatar = models.ImageField(default='media/default.jpg', upload_to='profile_images')
    bio = models.TextField()

    def __str__(self):
        return self.user.username

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)

        if self.avatar and os.path.exists(self.avatar.path):
            img = Image.open(self.avatar.path)
            if img.height > 100 or img.width > 100:
                img.thumbnail((100, 100))
                img.save(self.avatar.path)



class UserImageModel(models.Model):
    image = models.ImageField(upload_to='images/')
    label = models.CharField(max_length=20, default='data')

    def __str__(self):
        return str(self.image)

