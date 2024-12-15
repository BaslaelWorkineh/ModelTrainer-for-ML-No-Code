# Generated by Django 5.1.4 on 2024-12-15 14:28

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='ModelSelection',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('selected_model', models.CharField(max_length=255)),
                ('parameters', models.JSONField()),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name='UploadedFile',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('file', models.FileField(upload_to='uploads/')),
                ('headers', models.JSONField()),
                ('data', models.JSONField()),
                ('uploaded_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name='TrainingHistory',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('accuracy', models.FloatField()),
                ('loss', models.FloatField()),
                ('validation_accuracy', models.FloatField()),
                ('validation_loss', models.FloatField()),
                ('progress', models.IntegerField()),
                ('trained_at', models.DateTimeField(auto_now_add=True)),
                ('model_selection', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='train.modelselection')),
            ],
        ),
    ]
