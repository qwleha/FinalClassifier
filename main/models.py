from django.db import models

# Create your models here.

class Article(models.Model):
    href = models.CharField(max_length=256)
    title = models.TextField()
    text = models.TextField()
    type = models.CharField(max_length=32, default='')

    def __str__(self):
        return self.title
