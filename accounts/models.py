from django.contrib.auth.models import AbstractUser, BaseUserManager
from django.db import models
from django.db.models.signals import post_delete, pre_save
from django.dispatch import receiver
from django.contrib.admin.models import LogEntry
import json

class Institution(models.Model):
    name = models.CharField(max_length=100, unique=True)
    subdomain = models.CharField(max_length=100, unique=True)
    domain = models.CharField(max_length=253, unique=True, null=True, blank=True)
    allowed_domains = models.TextField(default='[]')
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

    def get_allowed_domains(self):
        return json.loads(self.allowed_domains)

    def set_allowed_domains(self, domains):
        self.allowed_domains = json.dumps(list(domains))

    def is_domain_allowed(self, domain):
        return domain in self.get_allowed_domains()

class TenantAwareModel(models.Model):
    institution = models.ForeignKey(Institution, on_delete=models.CASCADE, related_name='%(app_label)s_%(class)s_related')

    class Meta:
        abstract = True

class UserManager(BaseUserManager):
    def create_user(self, email, first_name, last_name, password=None, **extra_fields):
        if not email:
            raise ValueError('The Email field must be set')
        email = self.normalize_email(email)
        user = self.model(email=email, first_name=first_name, last_name=last_name, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, first_name, last_name, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        extra_fields.setdefault('is_active', True)
        extra_fields.setdefault('role', 'superadmin')
        if extra_fields.get('is_staff') is not True:
            raise ValueError('Superuser must have is_staff=True.')
        if extra_fields.get('is_superuser') is not True:
            raise ValueError('Superuser must have is_superuser=True.')
        return self.create_user(email, first_name, last_name, password, **extra_fields)

class User(AbstractUser):
    ROLE_CHOICES = (
        ('superadmin', 'Superadmin'),
        ('admin', 'Admin'),
        ('user', 'User'),
    )

    first_name = models.CharField(max_length=30)
    last_name = models.CharField(max_length=30)
    email = models.EmailField(unique=True)
    institution = models.ForeignKey(Institution, on_delete=models.SET_NULL, null=True, blank=True, related_name='users')
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='user')
    is_active = models.BooleanField(default=False)
    is_email_verified = models.BooleanField(default=False)

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['first_name', 'last_name']

    objects = UserManager()

    def save(self, *args, **kwargs):
        if not self.username:
            self.username = self.email
        if not self.institution:
            domain = self.email.split('@')[1]
            self.institution = next((inst for inst in Institution.objects.all() if inst.is_domain_allowed(domain)), None)
        super(User, self).save(*args, **kwargs)

    def __str__(self):
        return self.email

class Member(TenantAwareModel):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='account_member')
    institution = models.ForeignKey(Institution, on_delete=models.CASCADE, related_name='account_members')
    # Add other member-specific fields here

class AIModel(TenantAwareModel):
    name = models.CharField(max_length=100)
    file_path = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} - {self.institution.name}"

@receiver(post_delete, sender=User)
def delete_log_entries(sender, instance, **kwargs):
    LogEntry.objects.filter(object_id=instance.id).delete()

@receiver(pre_save, sender=User)
def update_log_entries(sender, instance, **kwargs):
    if instance.pk:
        previous_instance = User.objects.get(pk=instance.pk)
        if previous_instance.is_staff != instance.is_staff:
            LogEntry.objects.filter(object_id=instance.id).delete()