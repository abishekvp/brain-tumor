"""
convert_images_to_base64 management command

Usage:
    python manage.py convert_images_to_base64 [--dry-run] [--keep-files] [--no-backup]
"""

from django.core.management.base import BaseCommand
import convert_images_to_base64


class Command(BaseCommand):
    help = "Reads database image entries, converts them to Base64 data URIs, saves to DB, and deletes converted files from disk."

    def add_arguments(self, parser):
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Simulate the process without modifying the database or deleting files.",
        )
        parser.add_argument(
            "--keep-files",
            action="store_true",
            help="Convert images to Base64 in DB but do not delete files from disk.",
        )
        parser.add_argument(
            "--no-backup",
            action="store_true",
            help="Skip automatic database backup before conversion.",
        )

    def handle(self, *args, **options):
        convert_images_to_base64.process_entries(
            dry_run=options["dry_run"],
            keep_files=options["keep_files"],
            no_backup=options["no_backup"],
        )
