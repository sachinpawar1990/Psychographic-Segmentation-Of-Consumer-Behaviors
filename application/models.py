from django.db import models
from django.contrib.auth.models import Permission, User
from mongoengine import *


def user_directory_path(instance, filename):
#    return 'documents/user_{0}/new/{1}'.format(instance.user.id, filename)
    return 'application/static/application/images/user_{0}/new/{1}'.format(instance.user.id, filename)

class Upload(models.Model):
    docfile = models.FileField(upload_to=user_directory_path)
    filename = models.CharField(max_length=250)
    user = models.ForeignKey(User, default=2)
    date = models.DateTimeField()

    def __str__(self):
        return self.filename


class Table(DynamicDocument):
    username = IntField()
    fid = IntField()
    tablename = StringField(max_length=150)
