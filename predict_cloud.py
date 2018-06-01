from googleapiclient import discovery
import os

os.putenv('GOOGLE_APPLICATION_CREDENTIALS', '/home/anderson/.ssh/gcp')


def predict_json():
    service = discovery.build('ml', 'v1')


if __name__ == '__main__':
    predict_json()
