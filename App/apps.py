from django.apps import AppConfig
import sys

_SYNC_EXECUTED = False


def trigger_default_image_sync():
    global _SYNC_EXECUTED
    if _SYNC_EXECUTED:
        return
    _SYNC_EXECUTED = True
    try:
        from .models import sync_default_image_from_media
        sync_default_image_from_media()
    except Exception:
        pass


class AppConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "app"

    def ready(self):
        # Prevent executing database queries during schema migrations or inspection commands
        skip_commands = ["makemigrations", "migrate", "inspectdb", "showmigrations", "sqlmigrate"]
        if any(cmd in sys.argv for cmd in skip_commands):
            return

        from django.core.signals import request_started
        from django.db.models.signals import post_migrate

        request_started.connect(
            lambda **kwargs: trigger_default_image_sync(),
            dispatch_uid="app_sync_default_image_req"
        )
        post_migrate.connect(
            lambda **kwargs: trigger_default_image_sync(),
            dispatch_uid="app_sync_default_image_migrate"
        )

        # On runserver or testserver startup, trigger sync in a short background thread once apps are ready
        if any(cmd in sys.argv for cmd in ["runserver", "testserver"]):
            import threading
            threading.Thread(target=trigger_default_image_sync, daemon=True).start()


