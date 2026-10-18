from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import os

DEFAULT_AVATAR_FALLBACK = (
    "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8U"
    "HRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIy"
    "MjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCABkAEsDASIAAhEBAxEB/8QAHwAAAQUBAQEBA"
    "QEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0Kxw"
    "RVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJip"
    "KTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHw"
    "EAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoE"
    "IFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6"
    "goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3"
    "+Pn6/9oADAMBAAIRAxEAPwDwCilpKBhRRS4oASilooEJS0UUAFFFFAwooopgFJRRSEFLSUUALRSUtABRRRQAUlLSGgAoo"
    "ooAKKKKAClpKWgAooooAKQ0tFAxKKKKBBRRRQAUtJS0AFFFFAwooopgFFFFIBKKWigQUUUUDCiiimAUUUUAFFFFABRRRQ"
    "AUUUUAFFFLigBTSUUUAFFFFABiiiigAooooAKcKKKAP//Z"
)
DEFAULT_AVATAR_BASE64 = DEFAULT_AVATAR_FALLBACK


class Base64Image(str):
    """
    Subclass of str representing a Base64 image data URI.
    Provides .url and .path properties for full compatibility with existing
    Django templates ({{ obj.image.url }}, {{ user.profile.avatar.url }})
    and code expecting a file-like attribute.
    """
    @property
    def url(self):
        return str(self)

    @property
    def path(self):
        return ""


class Base64ImageField(models.TextField):
    """
    TextField that stores Base64 image data URIs directly in the database.
    """
    description = "A base64 encoded image string stored directly in the database."

    def from_db_value(self, value, expression, connection):
        if not value:
            return None
        return Base64Image(value)

    def to_python(self, value):
        if not value:
            return None
        if isinstance(value, Base64Image):
            return value
        return Base64Image(value)

    def get_prep_value(self, value):
        if value is None:
            return None
        return str(value)


class SystemAsset(models.Model):
    """
    Stores system-level static/default assets as Base64 data URIs directly in the database.
    Eliminates reliance on physical image files for fallbacks like default.jpg.
    """
    key = models.CharField(max_length=100, unique=True, primary_key=True)
    image = Base64ImageField()
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"SystemAsset({self.key})"


def get_system_asset(key: str, fallback: str = "") -> str:
    """Retrieve a system asset base64 string from the database with graceful fallback."""
    try:
        asset = SystemAsset.objects.filter(key=key).first()
        if asset and asset.image:
            return str(asset.image)
    except Exception:
        pass
    return fallback


def set_system_asset(key: str, base64_data: str) -> None:
    """Store or update a system asset base64 data URI in the database."""
    try:
        SystemAsset.objects.update_or_create(
            key=key,
            defaults={"image": base64_data}
        )
    except Exception as e:
        print(f"[WARN] Failed to save SystemAsset '{key}': {e}")


def get_default_avatar() -> str:
    """Returns the default avatar base64 data URI from the database (or fallback)."""
    return get_system_asset("default_avatar", fallback=DEFAULT_AVATAR_FALLBACK)


def sync_default_image_from_media():
    """
    Executed on server startup:
    Checks if a default image (default.jpg, default.jpeg, default.png, default.jpj)
    is provided in the media directory.
    If provided:
        - Reads and converts it to a Base64 data URI
        - Uploads/updates it in the database (SystemAsset key='default_avatar')
        - Deletes the physical file from media/
    If not provided:
        - Uses the existing default avatar stored in the database.
    """
    from django.conf import settings
    from pathlib import Path
    import base64

    media_root = Path(settings.MEDIA_ROOT).resolve()
    base_dir = Path(settings.BASE_DIR).resolve()

    search_dirs = [media_root, base_dir / "media"]
    file_patterns = ["default.jpg", "default.jpeg", "default.png", "default.jpj"]

    found_file = None
    for d in search_dirs:
        for pattern in file_patterns:
            candidate = d / pattern
            if candidate.is_file():
                found_file = candidate
                break
        if found_file:
            break

    if found_file and found_file.is_file():
        try:
            data = found_file.read_bytes()
            if len(data) > 0:
                mime = "image/jpeg"
                if data.startswith(b"\x89PNG"):
                    mime = "image/png"
                elif data.startswith(b"\xff\xd8\xff"):
                    mime = "image/jpeg"
                elif data.startswith(b"GIF8"):
                    mime = "image/gif"
                elif data.startswith(b"RIFF") and len(data) > 12 and data[8:12] == b"WEBP":
                    mime = "image/webp"

                b64_str = base64.b64encode(data).decode("utf-8")
                data_uri = f"data:{mime};base64,{b64_str}"

                # Upload to database
                set_system_asset("default_avatar", data_uri)
                print(f"[SERVER STARTUP] Found '{found_file.name}' in media: Uploaded to DB as 'default_avatar'.")

                # Delete the physical file from disk
                try:
                    found_file.unlink()
                    print(f"[SERVER STARTUP] Deleted converted '{found_file.name}' from media directory.")
                except OSError as e:
                    print(f"[SERVER STARTUP] Warning: Could not remove physical file '{found_file}': {e}")
        except Exception as e:
            print(f"[SERVER STARTUP] Error reading/uploading default image '{found_file}': {e}")
    else:
        # Not provided in media -> ensure existing record in DB
        try:
            existing = SystemAsset.objects.filter(key="default_avatar").first()
            if not existing or not existing.image:
                set_system_asset("default_avatar", DEFAULT_AVATAR_FALLBACK)
                print("[SERVER STARTUP] Initialized 'default_avatar' in database with fallback.")
        except Exception:
            pass


class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    avatar = Base64ImageField(default=get_default_avatar, blank=True, null=True)
    bio = models.TextField(blank=True, default='')

    def __str__(self):
        return self.user.username

    @property
    def avatar_url(self):
        return str(self.avatar) if self.avatar else get_default_avatar()


class UserImageModel(models.Model):
    image = Base64ImageField()
    highlighted_image = Base64ImageField(null=True, blank=True)
    label = models.CharField(max_length=20, default='data')
    confidence = models.FloatField(null=True, blank=True)
    tumor_area_pct = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"Scan #{self.id} ({self.label})"

    @property
    def image_url(self):
        return str(self.image) if self.image else ""

    @property
    def highlighted_image_url(self):
        return str(self.highlighted_image) if self.highlighted_image else ""
