# -*- coding: utf-8 -*-
"""
Created on Tue May  1 15:21:47 2018

@author: sorrav
"""

class Config:
    APP_NAME = 'myapp'
    SECRET_KEY = 'secret-key-of-myapp'
    ADMIN_NAME = 'administrator'

    AWS_DEFAULT_REGION = 'ap-northeast-2'
    
    STATIC_PREFIX_PATH = 'static'
    ALLOWED_IMAGE_FORMATS = ['jpg', 'jpeg', 'png', 'gif']
    MAX_IMAGE_SIZE = 5242880 # 5MB

    
class DevelopmentConfig(Config):
    DEBUG = True
    
    AWS_ACCESS_KEY_ID = 'aws-access-key-for-dev'
    AWS_SECERT_ACCESS_KEY = 'aws-secret-access-key-for-dev'
    AWS_S3_BUCKET_NAME = 'aws-s3-bucket-name-for-dev'
    
    DATABASE_URI = 'database-uri-for-dev'
    
