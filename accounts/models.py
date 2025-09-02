from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver


class UserProfile(models.Model):
    ROLE_CHOICES = (
        ('FREE', 'Free'),
        ('PREMIUM', 'Premium'),
    )
    
    user = models.OneToOneField(User, on_delete=models.CASCADE)  # link each user to one profile
    role = models.CharField(max_length=10, choices=ROLE_CHOICES, default='FREE')

    def __str__(self):
        return f"{self.user.username} - {self.role}"


# 🔔 Signals to auto-create and save profile
@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(user=instance)


@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    instance.userprofile.save()
