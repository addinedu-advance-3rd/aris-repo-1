# Generated by Django 5.1.6 on 2025-02-06 06:02

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('aris', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Order',
            fields=[
                ('order_num', models.CharField(max_length=10, primary_key=True, serialize=False)),
                ('flavor', models.CharField(max_length=10)),
                ('topping', models.JSONField()),
                ('datetime', models.DateTimeField()),
                ('customer_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='orders', to='aris.customer')),
            ],
        ),
    ]
