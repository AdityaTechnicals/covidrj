

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cov_cnn_web.settings')

application = get_asgi_application()
