# Generated by Django 5.0.2 on 2024-04-19 07:05

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mobileapp', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='register',
            name='age',
            field=models.CharField(max_length=50, null=True),
        ),
        migrations.AlterField(
            model_name='register',
            name='contact',
            field=models.CharField(max_length=50, null=True),
        ),
        migrations.AlterField(
            model_name='register',
            name='email',
            field=models.EmailField(max_length=50, null=True),
        ),
        migrations.AlterField(
            model_name='register',
            name='name',
            field=models.CharField(max_length=50, null=True),
        ),
        migrations.AlterField(
            model_name='register',
            name='password',
            field=models.CharField(max_length=50, null=True),
        ),
    ]